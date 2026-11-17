"""
05_deepsdf_implicit.py
======================
Neural Implicit Shape Representation with DeepSDF & Marching Cubes.

Key Concepts:
- Continuous Signed Distance Function f_theta(x, y, z, z_shape) -> s in R.
- Auto-Decoder paradigm: jointly optimizing MLP weights theta and per-shape latent codes z.
- Eikonal Regularization: ||grad_x f_theta(x)||_2 = 1.
- Surface Extraction via Marching Cubes on the zero-level set f_theta(x) = 0.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
from tqdm import tqdm

from utils import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_SAMPLES_DIR,
    ensure_dirs,
    evaluate_shape_sdf,
    get_device,
    plot_mesh_on_ax,
)


class DeepSDFMLP(nn.Module):
    """
    Coordinate-based MLP with skip connection (Park et al., CVPR 2019).
    Input: 3D coordinate (x, y, z) concatenated with shape latent code z.
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        in_dim = 3 + latent_dim
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.layer3 = nn.Sequential(nn.Linear(hidden_dim + in_dim, hidden_dim), nn.SiLU())
        self.layer4 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, coords: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        x_in = torch.cat([coords, latent], dim=-1)
        h = self.layer1(x_in)
        h = self.layer2(h)
        # Skip connection concatenating original input
        h = self.layer3(torch.cat([h, x_in], dim=-1))
        h = self.layer4(h)
        return self.out(h).squeeze(-1)


def sample_sdf_training_data(shape_names, samples_per_shape: int = 4096, seed: int = 42):
    """Sample 3D coordinates (dense near surface + uniform in [-1, 1]^3) and exact SDFs."""
    rng = np.random.default_rng(seed)
    all_coords, all_sdfs, all_ids = [], [], []

    for idx, name in enumerate(shape_names):
        n_uniform = samples_per_shape // 2
        n_near = samples_per_shape - n_uniform
        pts_u = rng.uniform(-1.0, 1.0, size=(n_uniform, 3)).astype(np.float32)

        # Concentrate half the samples near the surface for sharp reconstruction
        pts_cand = rng.uniform(-1.0, 1.0, size=(n_near * 8, 3)).astype(np.float32)
        sdf_cand = evaluate_shape_sdf(name, pts_cand)
        near_mask = np.abs(sdf_cand) < 0.15
        pts_n = pts_cand[near_mask][:n_near]
        if len(pts_n) < n_near:
            pad = rng.uniform(-1.0, 1.0, size=(n_near - len(pts_n), 3)).astype(np.float32)
            pts_n = np.concatenate([pts_n, pad], axis=0)

        pts = np.concatenate([pts_u, pts_n], axis=0)
        sdfs = evaluate_shape_sdf(name, pts).astype(np.float32)
        ids = np.full(len(pts), idx, dtype=np.int64)

        all_coords.append(pts)
        all_sdfs.append(sdfs)
        all_ids.append(ids)

    return (
        torch.from_numpy(np.concatenate(all_coords, axis=0)),
        torch.from_numpy(np.concatenate(all_sdfs, axis=0)),
        torch.from_numpy(np.concatenate(all_ids, axis=0)),
    )


def extract_mesh_marching_cubes(model: nn.Module, latent: torch.Tensor, grid_res: int = 40, device=None):
    """Evaluate SDF on a 3D grid and extract zero-level set mesh via Marching Cubes."""
    coords_1d = np.linspace(-1.0, 1.0, grid_res, dtype=np.float32)
    gx, gy, gz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    grid_pts = torch.from_numpy(np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)).to(device)

    sdfs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(grid_pts), 8192):
            chunk = grid_pts[i : i + 8192]
            z = latent.unsqueeze(0).expand(len(chunk), -1)
            sdfs.append(model(chunk, z).cpu())
    vol = torch.cat(sdfs, dim=0).numpy().reshape(grid_res, grid_res, grid_res)

    verts, faces, _, _ = marching_cubes(vol, level=0.0)
    # Map grid indices [0, grid_res - 1] back to [-1, 1]^3
    verts = (verts / (grid_res - 1)) * 2.0 - 1.0
    return verts, faces, vol


def main():
    parser = argparse.ArgumentParser(description="DeepSDF Implicit Representation & Marching Cubes")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = get_device(args.device)
    epochs = 5 if args.quick_test else args.epochs
    samples_per_shape = 1024 if args.quick_test else 6144
    shapes = ["sphere", "torus", "chair"]

    print(f"[05_deepsdf_implicit] Device: {device} | Shapes: {shapes} | Epochs: {epochs}")

    coords_all, sdfs_all, ids_all = sample_sdf_training_data(shapes, samples_per_shape=samples_per_shape)
    dataset = torch.utils.data.TensorDataset(coords_all, sdfs_all, ids_all)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = DeepSDFMLP(latent_dim=args.latent_dim, hidden_dim=128).to(device)
    # Auto-Decoder: per-shape learnable latent codes
    latents = nn.Embedding(len(shapes), args.latent_dim).to(device)
    nn.init.normal_(latents.weight, mean=0.0, std=0.01)

    optimizer = optim.Adam(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": latents.parameters(), "lr": args.lr},
        ]
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for pts, gt_sdf, shape_ids in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            pts = pts.to(device).requires_grad_(True)
            gt_sdf = gt_sdf.to(device)
            z = latents(shape_ids.to(device))

            optimizer.zero_grad()
            pred_sdf = model(pts, z)

            # Clamped L1 SDF loss + latent L2 prior + Eikonal gradient norm loss
            loss_sdf = torch.abs(torch.clamp(pred_sdf, -0.2, 0.2) - torch.clamp(gt_sdf, -0.2, 0.2)).mean()
            grad_pts = torch.autograd.grad(
                outputs=pred_sdf.sum(),
                inputs=pts,
                create_graph=True,
            )[0]
            loss_eikonal = ((grad_pts.norm(dim=-1) - 1.0) ** 2).mean()
            loss_reg = (z ** 2).mean()

            loss = loss_sdf + 0.02 * loss_eikonal + 1e-4 * loss_reg
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d} | Total Loss: {total_loss/len(loader):.5f}")

    torch.save(
        {"model": model.state_dict(), "latents": latents.state_dict()},
        os.path.join(DEFAULT_CHECKPOINT_DIR, "05_deepsdf.pt"),
    )

    # Extract meshes via Marching Cubes & plot 2D SDF cross-sections
    fig = plt.figure(figsize=(12, 8))
    for i, name in enumerate(shapes):
        z_i = latents(torch.tensor(i, device=device))
        try:
            verts, faces, vol = extract_mesh_marching_cubes(model, z_i, grid_res=40, device=device)
            ax3d = fig.add_subplot(2, 3, i + 1, projection="3d")
            plot_mesh_on_ax(ax3d, verts, faces, title=f"Marching Cubes: {name}", color="mediumaquamarine")
        except ValueError:
            # Fallback if quick-test hasn't formed zero-level set yet
            ax3d = fig.add_subplot(2, 3, i + 1)
            ax3d.text(0.5, 0.5, "No zero-level set\n(increase epochs)", ha="center", va="center")
            ax3d.set_title(f"{name} (no surface)")
            coords_1d = np.linspace(-1.0, 1.0, 40, dtype=np.float32)
            gx, gy, gz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
            grid_pts = torch.from_numpy(np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)).to(device)
            with torch.no_grad():
                vol = model(grid_pts, z_i.unsqueeze(0).expand(len(grid_pts), -1)).cpu().numpy().reshape(40, 40, 40)

        # 2D SDF slice (Y=0 plane -> index 20)
        ax2d = fig.add_subplot(2, 3, i + 4)
        im = ax2d.imshow(vol[:, 20, :].T, origin="lower", cmap="coolwarm", vmin=-0.5, vmax=0.5)
        ax2d.contour(vol[:, 20, :].T, levels=[0.0], colors="black", linewidths=1.5)
        ax2d.set_title(f"SDF Slice y=0 ({name})")
        ax2d.axis("off")

    out_path = os.path.join(DEFAULT_SAMPLES_DIR, "05_deepsdf_implicit.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
