"""
07_gaussian_splatting.py
========================
Differentiable 3D Gaussian Splatting (3DGS) in Pure PyTorch.

Key Concepts:
- Explicit Anisotropic 3D Gaussians: position mu, scale s, rotation quaternion q,
  opacity alpha, and RGB color c.
- 3D Covariance Parameterization: Sigma = R * S * S^T * R^T.
- EWA Splatting / Perspective Projection to 2D Screen Space:
    Sigma_2d = J * W * Sigma * W^T * J^T
- Depth-sorted Alpha-Blending Compositing (no MLP ray marching!).
"""

import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import DEFAULT_CHECKPOINT_DIR, DEFAULT_SAMPLES_DIR, ensure_dirs, get_device


def generate_synthetic_views(num_views: int = 16, img_size: int = 48, focal: float = 55.0):
    """Reuse analytic multi-view 3D scene for direct comparison with NeRF."""
    from importlib import import_module
    nerf_mod = import_module("06_tiny_nerf")
    return nerf_mod.generate_synthetic_multiview_scene(num_views=num_views, img_size=img_size, focal=focal)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert normalized quaternions (N, 4) [w, x, y, z] -> (N, 3, 3) rotation matrices."""
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=1,
    )


class GaussianModel3D(nn.Module):
    """Learnable collection of N anisotropic 3D Gaussians."""

    def __init__(self, num_gaussians: int = 250):
        super().__init__()
        self.means = nn.Parameter((torch.rand(num_gaussians, 3) * 2.0 - 1.0) * 0.8)
        self.log_scales = nn.Parameter(torch.full((num_gaussians, 3), math.log(0.08)))
        quats = torch.zeros(num_gaussians, 4)
        quats[:, 0] = 1.0
        self.quats = nn.Parameter(quats)
        self.logit_opacities = nn.Parameter(torch.zeros(num_gaussians))
        self.colors = nn.Parameter(torch.rand(num_gaussians, 3))

    def render(self, c2w: torch.Tensor, H: int, W: int, focal: float) -> torch.Tensor:
        """
        Differentiable 3D-to-2D Gaussian Splatting forward pass.
        Projects all 3D Gaussians onto the image plane and composites back-to-front.
        """
        device = self.means.device
        w2c = torch.inverse(c2w)
        R_cam = w2c[:3, :3]
        t_cam = w2c[:3, 3]

        # Transform Gaussian centers to camera coordinates (camera looks down -Z)
        pts_cam = self.means @ R_cam.T + t_cam  # (N, 3)
        z_depth = -pts_cam[:, 2]  # positive depth in front of camera
        valid = z_depth > 0.2

        pts_cam = pts_cam[valid]
        z_depth = z_depth[valid]
        
        opacities = torch.sigmoid(self.logit_opacities[valid])
        colors = torch.sigmoid(self.colors[valid])

        scales = torch.exp(self.log_scales[valid]).clamp(1e-3, 0.5)
        R_obj = quaternion_to_rotation_matrix(self.quats[valid])

        # 3D Covariance: Sigma = R * S * S^T * R^T
        S_mat = torch.diag_embed(scales)
        M = R_obj @ S_mat
        Sigma_3d = M @ M.transpose(-1, -2)  # (K, 3, 3)

        # Project to 2D image coordinates (u, v)
        u = focal * (pts_cam[:, 0] / z_depth) + W * 0.5
        v = -focal * (pts_cam[:, 1] / z_depth) + H * 0.5
        means_2d = torch.stack([u, v], dim=-1)  # (K, 2)

        # Jacobian of perspective projection: J (K, 2, 3)
        inv_z = 1.0 / z_depth
        inv_z2 = inv_z * inv_z
        zeros = torch.zeros_like(inv_z)
        J = torch.stack(
            [
                torch.stack([focal * inv_z, zeros, focal * pts_cam[:, 0] * inv_z2], dim=-1),
                torch.stack([zeros, -focal * inv_z, -focal * pts_cam[:, 1] * inv_z2], dim=-1),
            ],
            dim=1,
        )  # (K, 2, 3)

        # Projected 2D covariance: Sigma_2d = J * R_cam * Sigma_3d * R_cam^T * J^T + 0.3*I
        T_mat = J @ R_cam.unsqueeze(0)
        Sigma_2d = T_mat @ Sigma_3d @ T_mat.transpose(-1, -2)
        Sigma_2d = Sigma_2d + 0.3 * torch.eye(2, device=device).unsqueeze(0)

        # Sort Gaussians front-to-back by camera depth
        order = torch.argsort(z_depth, descending=False)
        means_2d = means_2d[order]
        Sigma_2d = Sigma_2d[order]
        opacities = opacities[order]
        colors = colors[order]

        inv_Sigma = torch.inverse(Sigma_2d)  # (K, 2, 2)

        # Pixel grid (H, W, 2)
        py, px = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing="ij",
        )
        grid = torch.stack([px, py], dim=-1).view(H * W, 1, 2)  # (HW, 1, 2)

        diff = grid - means_2d.unsqueeze(0)  # (HW, K, 2)
        mahalanobis = torch.einsum("pki,kij,pkj->pk", diff, inv_Sigma, diff)  # (HW, K)
        gauss_weight = torch.exp(-0.5 * mahalanobis.clamp(max=40.0))  # (HW, K)

        alpha = (opacities.unsqueeze(0) * gauss_weight).clamp(max=0.99)  # (HW, K)
        # Front-to-back transmittance T_k = prod_{j < k} (1 - alpha_j)
        ones = torch.ones((H * W, 1), device=device)
        T = torch.cumprod(torch.cat([ones, 1.0 - alpha[:, :-1] + 1e-7], dim=-1), dim=-1)
        weights = alpha * T  # (HW, K)

        img_flat = weights @ colors  # (HW, 3)
        bg_weight = 1.0 - weights.sum(dim=-1, keepdim=True)
        img_flat = img_flat + bg_weight * 0.96
        return img_flat.view(H, W, 3)


def main():
    parser = argparse.ArgumentParser(description="Differentiable 3D Gaussian Splatting in PyTorch")
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--num-gaussians", type=int, default=250)
    parser.add_argument("--img-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = get_device(args.device)
    iters = 30 if args.quick_test else args.iters

    print(
        f"[07_gaussian_splatting] Device: {device} | Gaussians: {args.num_gaussians} | Iters: {iters}"
    )

    images_np, poses_np, focal = generate_synthetic_views(num_views=16, img_size=args.img_size)
    images = torch.from_numpy(images_np).to(device)
    poses = torch.from_numpy(poses_np).to(device)

    test_indices = [0, 8]
    train_indices = [i for i in range(len(images_np)) if i not in test_indices]

    model = GaussianModel3D(num_gaussians=args.num_gaussians).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    H = W = args.img_size
    for step in tqdm(range(1, iters + 1), desc="Optimizing 3D Gaussians"):
        v_idx = train_indices[step % len(train_indices)]
        optimizer.zero_grad()
        pred_img = model.render(poses[v_idx], H, W, focal)
        loss = torch.abs(pred_img - images[v_idx]).mean()
        loss.backward()
        optimizer.step()

        if step % max(1, iters // 5) == 0 or step == iters:
            print(f"Iter {step:04d} | L1 Photometric Loss: {loss.item():.5f}")

    torch.save(model.state_dict(), os.path.join(DEFAULT_CHECKPOINT_DIR, "07_gaussian_splatting.pt"))

    model.eval()
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.5))
    with torch.no_grad():
        for row, v_idx in enumerate(test_indices):
            pred_img = model.render(poses[v_idx], H, W, focal).clamp(0.0, 1.0).cpu().numpy()
            axes[row, 0].imshow(images_np[v_idx])
            axes[row, 0].set_title(f"GT Novel View #{v_idx}")
            axes[row, 1].imshow(pred_img)
            axes[row, 1].set_title(f"3DGS Splat Render #{v_idx}")
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")

    out_path = os.path.join(DEFAULT_SAMPLES_DIR, "07_gaussian_splatting.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
