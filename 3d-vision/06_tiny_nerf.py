"""
06_tiny_nerf.py
===============
Neural Radiance Fields (NeRF) from Scratch in PyTorch.

Key Concepts:
- Continuous 5D Radiance Field f_theta(x, y, z, d) -> (RGB color, volume density sigma).
- Fourier Positional Encoding gamma(p) to overcome spectral bias of MLPs.
- Pinhole Camera Ray Generation (origins o, directions d).
- Differentiable Volume Rendering via Quadrature Ray Marching:
    C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i
- Self-supervision from 2D multi-view images alone (Photometric MSE Loss).
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


def generate_synthetic_multiview_scene(
    num_views: int = 24, img_size: int = 48, focal: float = 55.0
):
    """
    Analytically ray-traces a colorful 3D scene (two spheres + a cube) from
    orbiting camera poses to produce calibrated multi-view RGB images without
    external dataset downloads.
    """
    H = W = img_size
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
    dirs_cam = np.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], axis=-1)

    # Scene primitives: (center, radius, rgb_color)
    spheres = [
        (np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.55, np.array([0.92, 0.30, 0.25], dtype=np.float32)),
        (np.array([0.65, 0.25, 0.30], dtype=np.float32), 0.30, np.array([0.20, 0.72, 0.90], dtype=np.float32)),
        (np.array([-0.60, -0.25, 0.35], dtype=np.float32), 0.28, np.array([0.95, 0.78, 0.22], dtype=np.float32)),
    ]

    images = []
    poses = []
    cam_radius = 2.6

    for v in range(num_views):
        az = 2.0 * math.pi * v / num_views
        el = 0.35 * math.sin(2.0 * az)
        cam_pos = np.array(
            [
                cam_radius * math.cos(el) * math.sin(az),
                cam_radius * math.sin(el),
                cam_radius * math.cos(el) * math.cos(az),
            ],
            dtype=np.float32,
        )

        # Look-at camera-to-world rotation matrix
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up_world)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)

        # Camera looks along -Z in local frame
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = cam_pos

        rays_d = dirs_cam @ c2w[:3, :3].T
        rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)
        rays_o = np.broadcast_to(cam_pos, rays_d.shape)

        # Analytic sphere ray-intersection + Lambertian shading
        img = np.ones((H, W, 3), dtype=np.float32) * 0.96  # light background
        best_t = np.full((H, W), 1e6, dtype=np.float32)
        light_dir = np.array([0.5, 0.8, 0.6], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)

        for center, r, base_col in spheres:
            oc = rays_o - center
            b = np.sum(oc * rays_d, axis=-1)
            c_val = np.sum(oc * oc, axis=-1) - r * r
            disc = b * b - c_val
            hit = disc > 0.0
            t_hit = np.where(hit, -b - np.sqrt(np.maximum(disc, 0.0)), 1e6)
            closer = (t_hit > 0.0) & (t_hit < best_t)
            if np.any(closer):
                hit_pts = rays_o[closer] + t_hit[closer, None] * rays_d[closer]
                normals = (hit_pts - center) / r
                shade = 0.35 + 0.65 * np.maximum(0.0, np.sum(normals * light_dir, axis=-1, keepdims=True))
                img[closer] = np.clip(base_col * shade, 0.0, 1.0)
                best_t[closer] = t_hit[closer]

        images.append(img)
        poses.append(c2w)

    return np.array(images, dtype=np.float32), np.array(poses, dtype=np.float32), focal


def get_rays(H: int, W: int, focal: float, c2w: torch.Tensor):
    """Generate ray origins and normalized directions in world space."""
    device = c2w.device
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy",
    )
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], dim=-1)
    rays_d = dirs @ c2w[:3, :3].T
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[:3, 3].expand_as(rays_d)
    return rays_o, rays_d


class PositionalEncoding(nn.Module):
    """Fourier feature mapping: gamma(p) = [p, sin(2^k pi p), cos(2^k pi p)]."""

    def __init__(self, num_freqs: int = 6):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for f in self.freqs:
            out.append(torch.sin(f * x))
            out.append(torch.cos(f * x))
        return torch.cat(out, dim=-1)


class TinyNeRF(nn.Module):
    """Compact NeRF MLP predicting (RGB, sigma) from 3D pos + viewing direction."""

    def __init__(self, pos_freqs: int = 6, dir_freqs: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.enc_pos = PositionalEncoding(pos_freqs)
        self.enc_dir = PositionalEncoding(dir_freqs)

        pos_dim = 3 * (1 + 2 * pos_freqs)
        dir_dim = 3 * (1 + 2 * dir_freqs)

        self.pts_net = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.sigma_head = nn.Linear(hidden_dim, 1)
        self.color_net = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, pts: torch.Tensor, dirs: torch.Tensor):
        h = self.pts_net(self.enc_pos(pts))
        sigma = torch.relu(self.sigma_head(h)).squeeze(-1)
        rgb = torch.sigmoid(self.color_net(torch.cat([h, self.enc_dir(dirs)], dim=-1)))
        return rgb, sigma


def render_rays(
    model: nn.Module,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float = 1.2,
    far: float = 4.0,
    num_samples: int = 48,
    perturb: bool = True,
):
    """Differentiable volume rendering via discrete quadrature along rays."""
    N_rays = rays_o.shape[0]
    t_vals = torch.linspace(near, far, num_samples, device=rays_o.device)
    z_vals = t_vals.unsqueeze(0).expand(N_rays, num_samples).clone()
    if perturb:
        mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)
        z_vals = lower + (upper - lower) * torch.rand_like(z_vals)

    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)  # (N_rays, K, 3)
    dirs = rays_d.unsqueeze(1).expand_as(pts)

    rgb, sigma = model(pts.reshape(-1, 3), dirs.reshape(-1, 3))
    rgb = rgb.view(N_rays, num_samples, 3)
    sigma = sigma.view(N_rays, num_samples)

    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    deltas = torch.cat([deltas, torch.full((N_rays, 1), 1e10, device=rays_o.device)], dim=-1)

    alpha = 1.0 - torch.exp(-sigma * deltas)
    transmittance = torch.cumprod(
        torch.cat([torch.ones((N_rays, 1), device=rays_o.device), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[:, :-1]
    weights = alpha * transmittance

    rgb_map = (weights.unsqueeze(-1) * rgb).sum(dim=1)
    # White/light background compositing
    acc_alpha = weights.sum(dim=1, keepdim=True)
    rgb_map = rgb_map + (1.0 - acc_alpha) * 0.96
    depth_map = (weights * z_vals).sum(dim=1)
    return rgb_map, depth_map


def main():
    parser = argparse.ArgumentParser(description="Tiny NeRF from Scratch in PyTorch")
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--img-size", type=int, default=48)
    parser.add_argument("--rays-per-iter", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = get_device(args.device)
    iters = 40 if args.quick_test else args.iters

    print(f"[06_tiny_nerf] Device: {device} | Image Size: {args.img_size}x{args.img_size} | Iters: {iters}")

    images_np, poses_np, focal = generate_synthetic_multiview_scene(num_views=24, img_size=args.img_size)
    images = torch.from_numpy(images_np).to(device)
    poses = torch.from_numpy(poses_np).to(device)

    # Hold out view 0 & 12 for novel view synthesis evaluation
    test_indices = [0, 12]
    train_indices = [i for i in range(len(images_np)) if i not in test_indices]

    model = TinyNeRF(pos_freqs=6, dir_freqs=4, hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    H = W = args.img_size
    for step in tqdm(range(1, iters + 1), desc="Training TinyNeRF"):
        view_idx = train_indices[step % len(train_indices)]
        rays_o, rays_d = get_rays(H, W, focal, poses[view_idx])
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        target_rgb = images[view_idx].reshape(-1, 3)

        sel = torch.randint(0, H * W, (args.rays_per_iter,), device=device)
        optimizer.zero_grad()
        pred_rgb, _ = render_rays(model, rays_o[sel], rays_d[sel], perturb=True)
        loss = ((pred_rgb - target_rgb[sel]) ** 2).mean()
        loss.backward()
        optimizer.step()

        if step % max(1, iters // 5) == 0 or step == iters:
            psnr = -10.0 * math.log10(max(loss.item(), 1e-10))
            print(f"Iter {step:04d} | Photometric MSE: {loss.item():.5f} | PSNR: {psnr:.2f} dB")

    torch.save(model.state_dict(), os.path.join(DEFAULT_CHECKPOINT_DIR, "06_tiny_nerf.pt"))

    # Render Novel Views (Held-out views 0 and 12)
    model.eval()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.8))
    with torch.no_grad():
        for row, v_idx in enumerate(test_indices):
            rays_o, rays_d = get_rays(H, W, focal, poses[v_idx])
            pred_rgb, pred_depth = render_rays(
                model, rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), perturb=False
            )
            pred_img = pred_rgb.view(H, W, 3).clamp(0.0, 1.0).cpu().numpy()
            depth_img = pred_depth.view(H, W).cpu().numpy()

            axes[row, 0].imshow(images_np[v_idx])
            axes[row, 0].set_title(f"GT Novel View #{v_idx}")
            axes[row, 1].imshow(pred_img)
            axes[row, 1].set_title(f"NeRF Rendered #{v_idx}")
            axes[row, 2].imshow(depth_img, cmap="magma")
            axes[row, 2].set_title(f"Expected Depth #{v_idx}")
            for c in range(3):
                axes[row, c].axis("off")

    out_path = os.path.join(DEFAULT_SAMPLES_DIR, "06_tiny_nerf.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
