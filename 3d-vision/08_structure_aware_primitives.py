"""
08_structure_aware_primitives.py
================================
Structure-Aware 3D Representation: Part Sets & Geometric Relationships.

Key Concepts:
- Element Geometry vs. Element Structure (part decomposition + relations).
- Representing a complex shape (chair) as a Set of Differentiable Box Primitives
  (seat, backrest, 4 legs) parameterized by centers c_k and half-extents s_k.
- Compositional CSG Union SDF: d(x) = min_k SDF_box(x - c_k; s_k).
- Structural Relationship Losses:
  1. Surface Chamfer Distance / SDF alignment to target shape.
  2. Bilateral Symmetry constraint across X=0 (left/right legs).
  3. Non-floating / Support Connectivity constraint between seat & legs/backrest.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm

from utils import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_SAMPLES_DIR,
    ensure_dirs,
    evaluate_shape_sdf,
    get_device,
    plot_pointcloud_on_ax,
    sample_surface_points,
)


def torch_sdf_box(points: torch.Tensor, center: torch.Tensor, half_extents: torch.Tensor) -> torch.Tensor:
    """Differentiable exact SDF of an axis-aligned box."""
    q = torch.abs(points - center.unsqueeze(0)) - half_extents.unsqueeze(0)
    outside = torch.norm(torch.clamp(q, min=0.0), dim=-1)
    inside = torch.clamp(torch.max(q, dim=-1)[0], max=0.0)
    return outside + inside


class PartSetShapeModel(nn.Module):
    """
    Represents a 3D object as K structured volumetric box primitives:
    0: Seat, 1: Backrest, 2-5: Four Legs.
    """

    def __init__(self):
        super().__init__()
        # Initial perturbed guesses near origin
        init_centers = torch.tensor(
            [
                [0.0, 0.0, 0.0],     # seat
                [0.0, 0.25, -0.25],  # backrest
                [-0.25, -0.3, -0.25],
                [0.25, -0.3, -0.25],
                [-0.25, -0.3, 0.25],
                [0.25, -0.3, 0.25],
            ],
            dtype=torch.float32,
        )
        init_log_extents = torch.log(
            torch.tensor(
                [
                    [0.30, 0.08, 0.30],
                    [0.30, 0.25, 0.08],
                    [0.08, 0.25, 0.08],
                    [0.08, 0.25, 0.08],
                    [0.08, 0.25, 0.08],
                    [0.08, 0.25, 0.08],
                ],
                dtype=torch.float32,
            )
        )
        self.centers = nn.Parameter(init_centers)
        self.log_extents = nn.Parameter(init_log_extents)

    def get_extents(self) -> torch.Tensor:
        return torch.exp(self.log_extents).clamp(0.03, 0.6)

    def forward_sdf(self, points: torch.Tensor) -> torch.Tensor:
        """CSG Union of all K part primitives: min_k SDF_k(x)."""
        extents = self.get_extents()
        part_sdfs = [torch_sdf_box(points, self.centers[k], extents[k]) for k in range(len(self.centers))]
        stacked = torch.stack(part_sdfs, dim=-1)  # (N, K)
        return stacked.min(dim=-1)[0]

    def structural_relationship_loss(self) -> torch.Tensor:
        """Enforce bilateral symmetry (X-axis) for legs & seat-to-leg contact."""
        extents = self.get_extents()
        # Left-right leg symmetry: (2 vs 3) and (4 vs 5)
        sym_23 = ((self.centers[2] - torch.tensor([-1.0, 1.0, 1.0], device=self.centers.device) * self.centers[3]) ** 2).sum()
        sym_45 = ((self.centers[4] - torch.tensor([-1.0, 1.0, 1.0], device=self.centers.device) * self.centers[5]) ** 2).sum()
        ext_sym = ((extents[2] - extents[3]) ** 2).sum() + ((extents[4] - extents[5]) ** 2).sum()

        # Legs top should touch bottom of seat (avoid floating parts)
        seat_bottom_y = self.centers[0, 1] - extents[0, 1]
        leg_tops_y = self.centers[2:, 1] + extents[2:, 1]
        contact_loss = ((leg_tops_y - seat_bottom_y) ** 2).mean()
        return sym_23 + sym_45 + ext_sym + contact_loss


def draw_box_on_ax(ax, center: np.ndarray, half_ext: np.ndarray, color: str, alpha: float = 0.75):
    """Draw a 3D box primitive on a matplotlib 3D axis."""
    cx, cy, cz = center
    hx, hy, hz = half_ext
    corners = np.array(
        [
            [cx - hx, cz - hz, cy - hy],
            [cx + hx, cz - hz, cy - hy],
            [cx + hx, cz + hz, cy - hy],
            [cx - hx, cz + hz, cy - hy],
            [cx - hx, cz - hz, cy + hy],
            [cx + hx, cz - hz, cy + hy],
            [cx + hx, cz + hz, cy + hy],
            [cx - hx, cz + hz, cy + hy],
        ]
    )
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],
        [corners[4], corners[5], corners[6], corners[7]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[2], corners[3], corners[7], corners[6]],
        [corners[1], corners[2], corners[6], corners[5]],
        [corners[0], corners[3], corners[7], corners[4]],
    ]
    poly = Poly3DCollection(faces, facecolors=color, linewidths=0.6, edgecolors="k", alpha=alpha)
    ax.add_collection3d(poly)


def main():
    parser = argparse.ArgumentParser(description="Structure-Aware Part Primitives Optimization")
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--lr", type=float, default=1.5e-2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = get_device(args.device)
    steps = 35 if args.quick_test else args.steps

    print(f"[08_structure_aware_primitives] Device: {device} | Target: chair | Steps: {steps}")

    rng = np.random.default_rng(42)
    pts_np = rng.uniform(-0.85, 0.85, size=(4096, 3)).astype(np.float32)
    gt_sdf_np = evaluate_shape_sdf("chair", pts_np).astype(np.float32)
    pts_t = torch.tensor(pts_np, device=device)
    gt_sdf_t = torch.tensor(gt_sdf_np, device=device)

    target_surface = sample_surface_points("chair", num_points=1200, rng=rng)

    model = PartSetShapeModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for step in tqdm(range(1, steps + 1), desc="Fitting Part Primitives"):
        optimizer.zero_grad()
        pred_sdf = model.forward_sdf(pts_t)
        loss_sdf = torch.abs(torch.clamp(pred_sdf, -0.2, 0.2) - torch.clamp(gt_sdf_t, -0.2, 0.2)).mean()
        loss_struct = model.structural_relationship_loss()
        loss = loss_sdf + 0.15 * loss_struct
        loss.backward()
        optimizer.step()

        if step % max(1, steps // 5) == 0 or step == steps:
            print(
                f"Step {step:04d} | SDF Loss: {loss_sdf.item():.5f} | Struct Loss: {loss_struct.item():.5f}"
            )

    torch.save(model.state_dict(), os.path.join(DEFAULT_CHECKPOINT_DIR, "08_structure_aware.pt"))

    centers_np = model.centers.detach().cpu().numpy()
    extents_np = model.get_extents().detach().cpu().numpy()
    part_colors = ["tomato", "royalblue", "seagreen", "seagreen", "goldenrod", "goldenrod"]

    fig = plt.figure(figsize=(10, 4.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_pointcloud_on_ax(ax1, target_surface, title="Target Shape (Chair)", s=3.5)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for k in range(len(centers_np)):
        draw_box_on_ax(ax2, centers_np[k], extents_np[k], color=part_colors[k])
    ax2.set_xlim(-0.8, 0.8)
    ax2.set_ylim(-0.8, 0.8)
    ax2.set_zlim(-0.8, 0.8)
    ax2.set_title("Recovered Part Primitives (6 Parts)", fontsize=10)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])

    out_path = os.path.join(DEFAULT_SAMPLES_DIR, "08_structure_aware_primitives.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
