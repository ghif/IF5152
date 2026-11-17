"""
04_mesh_deformation.py
======================
Triangle Mesh Deformation (Pixel2Mesh / Mesh R-CNN Style).

Key Concepts:
- Explicit Triangle Mesh: Vertices V (N x 3) and Fixed Topology Faces F (M x 3).
- Deforming a spherical template mesh to fit a target 3D shape.
- Differentiable Mesh Losses:
  1. Surface Chamfer Distance (sampling points uniformly on mesh triangles).
  2. Edge Length Regularization (prevents degenerate/stretched triangles).
  3. Laplacian Smoothing (encourages local surface smoothness & prevents self-intersections).
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
    create_icosphere,
    ensure_dirs,
    get_device,
    plot_mesh_on_ax,
    plot_pointcloud_on_ax,
    sample_surface_points,
)


def extract_edges_and_neighbors(faces: np.ndarray, num_verts: int):
    """Build unique undirected edge list (E, 2) and vertex adjacency matrix."""
    edge_set = set()
    adj = [[] for _ in range(num_verts)]
    for f in faces:
        for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            u, v = int(min(a, b)), int(max(a, b))
            if (u, v) not in edge_set:
                edge_set.add((u, v))
                adj[u].append(v)
                adj[v].append(u)
    edges = np.array(sorted(edge_set), dtype=np.int64)
    return edges, adj


def sample_points_from_mesh(verts: torch.Tensor, faces: torch.Tensor, num_samples: int = 1024) -> torch.Tensor:
    """
    Differentiable area-weighted uniform surface sampling from a triangle mesh.
    Args:
        verts: (V, 3)
        faces: (F, 3)
    Returns:
        sampled_pts: (num_samples, 3)
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1) + 1e-12
    probs = areas / areas.sum()

    face_idx = torch.multinomial(probs, num_samples, replacement=True)
    # Uniform barycentric coordinates
    r1 = torch.sqrt(torch.rand(num_samples, 1, device=verts.device))
    r2 = torch.rand(num_samples, 1, device=verts.device)
    w0 = 1.0 - r1
    w1 = r1 * (1.0 - r2)
    w2 = r1 * r2

    return w0 * v0[face_idx] + w1 * v1[face_idx] + w2 * v2[face_idx]


def chamfer_distance_single(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(p1.unsqueeze(0), p2.unsqueeze(0), p=2.0).squeeze(0) ** 2
    return dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()


def edge_length_loss(verts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    diff = verts[edges[:, 0]] - verts[edges[:, 1]]
    return (diff ** 2).sum(dim=-1).mean()


def laplacian_smoothing_loss(verts: torch.Tensor, lap_matrix: torch.Tensor) -> torch.Tensor:
    """Uniform umbrella Laplacian: L @ V measures deviation from neighbor centroid."""
    lap = torch.sparse.mm(lap_matrix, verts)
    return (lap ** 2).sum(dim=-1).mean()


def build_uniform_laplacian(num_verts: int, edges: np.ndarray, device: torch.device) -> torch.Tensor:
    """Construct sparse N x N uniform Laplacian matrix L = I - D^{-1} A."""
    deg = np.zeros(num_verts, dtype=np.float32)
    rows, cols = [], []
    for u, v in edges:
        rows.extend([u, v])
        cols.extend([v, u])
        deg[u] += 1.0
        deg[v] += 1.0

    vals = [-1.0 / max(deg[r], 1.0) for r in rows]
    for i in range(num_verts):
        rows.append(i)
        cols.append(i)
        vals.append(1.0)

    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.tensor(vals, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, (num_verts, num_verts)).coalesce()


def main():
    parser = argparse.ArgumentParser(description="Template Mesh Deformation with Geometric Regularizers")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--target", type=str, default="cube", choices=["cube", "cylinder", "chair"])
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = get_device(args.device)
    steps = 30 if args.quick_test else args.steps

    print(f"[04_mesh_deformation] Device: {device} | Target: {args.target} | Steps: {steps}")

    # 1. Initialize template icosphere mesh
    init_verts, faces_np = create_icosphere(subdivisions=2, radius=0.75)
    edges_np, _ = extract_edges_and_neighbors(faces_np, len(init_verts))
    lap_matrix = build_uniform_laplacian(len(init_verts), edges_np, device)

    verts_template = torch.tensor(init_verts, dtype=torch.float32, device=device)
    faces_t = torch.tensor(faces_np, dtype=torch.long, device=device)
    edges_t = torch.tensor(edges_np, dtype=torch.long, device=device)

    # Learnable vertex offsets delta_V
    offsets = nn.Parameter(torch.zeros_like(verts_template))
    optimizer = optim.Adam([offsets], lr=args.lr)

    # 2. Target point cloud sampled from ground truth SDF surface
    rng = np.random.default_rng(42)
    target_pts_np = sample_surface_points(args.target, num_points=1500, rng=rng)
    target_pts = torch.tensor(target_pts_np, dtype=torch.float32, device=device)

    snapshots = []
    snap_steps = [0, steps // 3, (2 * steps) // 3, steps]

    for step in tqdm(range(steps + 1), desc="Deforming Mesh"):
        curr_verts = verts_template + offsets
        if step in snap_steps:
            snapshots.append((step, curr_verts.detach().cpu().numpy()))
        if step == steps:
            break

        optimizer.zero_grad()
        sampled_mesh_pts = sample_points_from_mesh(curr_verts, faces_t, num_samples=1024)
        loss_cd = chamfer_distance_single(sampled_mesh_pts, target_pts)
        loss_edge = edge_length_loss(curr_verts, edges_t)
        loss_lap = laplacian_smoothing_loss(curr_verts, lap_matrix)

        loss = loss_cd + 0.05 * loss_edge + 0.2 * loss_lap
        loss.backward()
        optimizer.step()

    print(
        f"Final Losses -> Chamfer: {loss_cd.item():.5f} | Edge: {loss_edge.item():.5f} | Lap: {loss_lap.item():.5f}"
    )

    torch.save(
        {"verts": (verts_template + offsets).detach().cpu(), "faces": faces_t.cpu()},
        os.path.join(DEFAULT_CHECKPOINT_DIR, "04_deformed_mesh.pt"),
    )

    # Visualization: Target Point Cloud + 4 Deformation Snapshots
    fig = plt.figure(figsize=(15, 3.8))
    ax0 = fig.add_subplot(1, 5, 1, projection="3d")
    plot_pointcloud_on_ax(ax0, target_pts_np, title=f"Target ({args.target})", s=3.0)

    for idx, (s, v_np) in enumerate(snapshots[:4]):
        ax = fig.add_subplot(1, 5, idx + 2, projection="3d")
        plot_mesh_on_ax(ax, v_np, faces_np, title=f"Step {s}")

    out_path = os.path.join(DEFAULT_SAMPLES_DIR, "04_mesh_deformation.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
