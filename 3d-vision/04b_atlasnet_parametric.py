"""
04b_atlasnet_parametric.py
==========================
Parametric Surface Decoding with AtlasNet (Groueix et al., CVPR 2018).

Key Concepts:
- Representing a 3D surface as a collection of K learned 2D-to-3D parametric patches:
    f_k(u, v; z) -> (x, y, z) for (u, v) in [0, 1]^2.
- Enforces local surface smoothness via continuous 2D manifold parameterization.
- Extracts explicit triangulated mesh patches directly from regular UV grids.
- Trained end-to-end with Chamfer Distance against target surface point clouds.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_SAMPLES_DIR,
    ensure_dirs,
    get_device,
    plot_mesh_on_ax,
    plot_pointcloud_on_ax,
    sample_surface_points,
)


def chamfer_distance_single(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(p1.unsqueeze(0), p2.unsqueeze(0), p=2.0).squeeze(0) ** 2
    return dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()


def create_uv_grid_topology(grid_res: int = 12):
    """Create a regular (grid_res x grid_res) UV grid in [0, 1]^2 and its triangle faces."""
    u = np.linspace(0.0, 1.0, grid_res, dtype=np.float32)
    v = np.linspace(0.0, 1.0, grid_res, dtype=np.float32)
    gu, gv = np.meshgrid(u, v, indexing="ij")
    uv_pts = np.stack([gu.reshape(-1), gv.reshape(-1)], axis=-1)

    faces = []
    for i in range(grid_res - 1):
        for j in range(grid_res - 1):
            idx = i * grid_res + j
            faces.append([idx, idx + 1, idx + grid_res])
            faces.append([idx + 1, idx + grid_res + 1, idx + grid_res])
    return uv_pts, np.array(faces, dtype=np.int64)


class AtlasPatchMLP(nn.Module):
    """Maps 2D UV coordinates + shape latent code z -> 3D surface point."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, uv: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z_exp = z.unsqueeze(0).expand(uv.shape[0], -1)
        return self.net(torch.cat([uv, z_exp], dim=-1))


class AtlasNetDecoder(nn.Module):
    """Atlas of K parametric surface patches covering a 3D shape."""

    def __init__(self, num_patches: int = 8, latent_dim: int = 32):
        super().__init__()
        self.patches = nn.ModuleList([AtlasPatchMLP(latent_dim=latent_dim) for _ in range(num_patches)])

    def forward(self, uv: torch.Tensor, z: torch.Tensor):
        return [patch(uv, z) for patch in self.patches]


def main():
    parser = argparse.ArgumentParser(description="AtlasNet Parametric Surface Decoding")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--num-patches", type=int, default=8)
    parser.add_argument("--target", type=str, default="torus", choices=["sphere", "cube", "torus", "chair"])
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = get_device(args.device)
    steps = 40 if args.quick_test else args.steps

    print(
        f"[04b_atlasnet_parametric] Device: {device} | Target: {args.target} "
        f"| Patches: {args.num_patches} | Steps: {steps}"
    )

    rng = np.random.default_rng(42)
    target_pts_np = sample_surface_points(args.target, num_points=1500, rng=rng)
    target_pts = torch.tensor(target_pts_np, dtype=torch.float32, device=device)

    latent_dim = 32
    z = nn.Parameter(torch.randn(latent_dim, device=device) * 0.1)
    model = AtlasNetDecoder(num_patches=args.num_patches, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(list(model.parameters()) + [z], lr=args.lr)

    uv_grid_np, patch_faces = create_uv_grid_topology(grid_res=12)
    uv_grid = torch.tensor(uv_grid_np, dtype=torch.float32, device=device)

    for step in tqdm(range(1, steps + 1), desc="Optimizing AtlasNet"):
        optimizer.zero_grad()
        # Sample random UV points per patch each iteration
        uv_rand = torch.rand(128, 2, device=device)
        patch_pts = model(uv_rand, z)
        all_pts = torch.cat(patch_pts, dim=0)
        loss = chamfer_distance_single(all_pts, target_pts)
        loss.backward()
        optimizer.step()

        if step % max(1, steps // 5) == 0 or step == steps:
            print(f"Step {step:04d} | Chamfer Loss: {loss.item():.5f}")

    torch.save(
        {"model": model.state_dict(), "z": z.detach().cpu()},
        os.path.join(DEFAULT_CHECKPOINT_DIR, "04b_atlasnet.pt"),
    )

    # Extract explicit multi-patch mesh from regular UV grid
    model.eval()
    with torch.no_grad():
        grid_patches = [p.cpu().numpy() for p in model(uv_grid, z)]

    # Combine all patch meshes into one (V, F) for visualization
    all_verts = []
    all_faces = []
    offset = 0
    for p_verts in grid_patches:
        all_verts.append(p_verts)
        all_faces.append(patch_faces + offset)
        offset += len(p_verts)
    combined_verts = np.concatenate(all_verts, axis=0)
    combined_faces = np.concatenate(all_faces, axis=0)
    combined_pts = combined_verts

    fig = plt.figure(figsize=(12, 4.2))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    plot_pointcloud_on_ax(ax1, target_pts_np, title=f"Target ({args.target})", s=3.0)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    plot_pointcloud_on_ax(ax2, combined_pts, title=f"AtlasNet Points ({args.num_patches} patches)", s=4.0)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    plot_mesh_on_ax(ax3, combined_verts, combined_faces, title="AtlasNet Extracted Mesh", color="gold")

    out_path = os.path.join(DEFAULT_SAMPLES_DIR, "04b_atlasnet_parametric.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
