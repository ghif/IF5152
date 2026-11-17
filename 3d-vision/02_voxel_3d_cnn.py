"""
02_voxel_3d_cnn.py
==================
Volumetric 3D Deep Learning with 3D Convolutional Neural Networks (3D CNNs).

Key Concepts:
- Explicit 3D Voxel Grids (V x V x V occupancy tensors).
- 3D Shape Classification via nn.Conv3d + Global Pooling.
- Single-View 2D-to-3D Voxel Reconstruction (2D CNN Encoder -> 3D Transposed Conv Decoder).
- Memory scaling trade-off: O(V^3) cubic growth.
"""

import argparse
import os
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_SAMPLES_DIR,
    MODELNET10_CATEGORIES,
    SHAPE_NAMES,
    download_and_extract_modelnet10,
    ensure_dirs,
    generate_voxel_grid,
    get_device,
    load_off_mesh,
    plot_voxels_on_ax,
    voxelize_mesh,
)


class ModelNet10VoxelDataset(Dataset):
    """
    ModelNet10 CAD dataset voxelized into V x V x V binary occupancy grids
    and 2D orthographic depth/silhouette projections.
    Pre-processes and caches .pt tensors into data/ModelNet10/cache/ for instant loading.
    """

    def __init__(
        self,
        split: str = "train",
        resolution: int = 24,
        categories: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        data_dir: Optional[str] = None,
    ):
        self.resolution = resolution
        self.split = split
        modelnet_dir = download_and_extract_modelnet10(data_dir)

        if categories is None or len(categories) == 0:
            self.categories = MODELNET10_CATEGORIES
        else:
            self.categories = [c.strip().lower() for c in categories if c.strip().lower() in MODELNET10_CATEGORIES]
            if not self.categories:
                self.categories = MODELNET10_CATEGORIES

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.categories)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        cache_dir = os.path.join(modelnet_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cat_key = "_".join(sorted(self.categories))
        if len(self.categories) == len(MODELNET10_CATEGORIES):
            cat_key = "all10"
        limit_str = f"_lim{max_samples}" if max_samples else ""
        cache_file = os.path.join(
            cache_dir,
            f"modelnet10_voxel_{split}_{cat_key}_res{resolution}{limit_str}.pt",
        )

        if os.path.exists(cache_file):
            print(f"[ModelNet10] Loading cached voxel dataset from {cache_file}...")
            cached = torch.load(cache_file, weights_only=False)
            self.voxels = cached["voxels"]
            self.projections = cached["projections"]
            self.labels = cached["labels"]
            return

        # Discover OFF files
        file_list = []
        for cat in self.categories:
            split_dir = os.path.join(modelnet_dir, cat, split)
            if not os.path.isdir(split_dir):
                continue
            cat_files = sorted([
                os.path.join(split_dir, f)
                for f in os.listdir(split_dir)
                if f.endswith(".off")
            ])
            if max_samples:
                cat_files = cat_files[:max_samples]
            for f in cat_files:
                file_list.append((f, self.class_to_idx[cat]))

        print(f"[ModelNet10] Voxelizing {len(file_list)} {split} meshes at {resolution}^3...")
        vox_list, proj_list, lbl_list = [], [], []
        rng = np.random.default_rng(42)

        for filepath, lbl in tqdm(file_list, desc=f"Voxelizing {split}", leave=False):
            try:
                verts, faces = load_off_mesh(filepath)
                grid, proj = voxelize_mesh(verts, faces, resolution=resolution, rng=rng)
                vox_list.append(grid[None, ...])  # (1, V, V, V)
                proj_list.append(proj)           # (1, V, V)
                lbl_list.append(lbl)
            except Exception as e:
                print(f"[Warning] Failed to voxelize {filepath}: {e}")

        self.voxels = torch.from_numpy(np.array(vox_list, dtype=np.float32))
        self.projections = torch.from_numpy(np.array(proj_list, dtype=np.float32))
        self.labels = torch.tensor(lbl_list, dtype=torch.long)

        print(f"[ModelNet10] Caching {len(self.labels)} voxel grids to {cache_file}...")
        torch.save(
            {
                "voxels": self.voxels,
                "projections": self.projections,
                "labels": self.labels,
            },
            cache_file,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.projections[idx], self.voxels[idx], self.labels[idx]


class SyntheticVoxelDataset(Dataset):
    """
    Generates random 3D voxel grids (with random scaling/shifts) and their
    orthographic 2D silhouette/depth projections for single-view 3D reconstruction.
    """

    def __init__(self, num_samples: int = 200, resolution: int = 24, seed: int = 42):
        self.resolution = resolution
        rng = np.random.default_rng(seed)

        base_voxels = {name: generate_voxel_grid(name, resolution) for name in SHAPE_NAMES}
        self.voxels = []
        self.projections = []
        self.labels = []

        for _ in range(num_samples):
            cls_idx = int(rng.integers(0, len(SHAPE_NAMES)))
            name = SHAPE_NAMES[cls_idx]
            vox = base_voxels[name].copy()

            # Random 3D axis flips / 90-degree rotations for augmentation
            k = int(rng.integers(0, 4))
            vox = np.rot90(vox, k=k, axes=(0, 2)).copy()
            if rng.random() > 0.5:
                vox = np.flip(vox, axis=0).copy()

            # Orthographic depth/silhouette projection along Z-axis (2D input)
            depth_proj = vox.mean(axis=2, keepdims=True)  # (V, V, 1)
            self.voxels.append(vox[None, ...])  # (1, V, V, V)
            self.projections.append(depth_proj.transpose(2, 0, 1))  # (1, V, V)
            self.labels.append(cls_idx)

        self.voxels = torch.from_numpy(np.array(self.voxels, dtype=np.float32))
        self.projections = torch.from_numpy(np.array(self.projections, dtype=np.float32))
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.projections[idx], self.voxels[idx], self.labels[idx]


class VoxelClassifier3D(nn.Module):
    """3D CNN Classifier operating directly on V x V x V occupancy grids."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.classifier(h)


class SingleViewVoxelReconstructor(nn.Module):
    """
    2D-to-3D Reconstruction Network:
    Encodes a 2D projection into a latent code z, then decodes via 3D transposed
    convolutions into a 24 x 24 x 24 voxel occupancy grid.
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder2d = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 24 -> 12
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 12 -> 6
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 6 -> 3
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_to_3d = nn.Linear(latent_dim, 64 * 3 * 3 * 3)
        self.decoder3d = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),  # 3 -> 6
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),  # 6 -> 12
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),   # 12 -> 24
        )

    def forward(self, img2d: torch.Tensor) -> torch.Tensor:
        z = self.encoder2d(img2d)
        h = torch.relu(self.fc_to_3d(z)).view(-1, 64, 3, 3, 3)
        return self.decoder3d(h)


def compute_voxel_iou(pred_logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    preds = (torch.sigmoid(pred_logits) > thresh).float()
    intersection = (preds * target).sum(dim=(1, 2, 3, 4))
    union = ((preds + target) > 0).float().sum(dim=(1, 2, 3, 4)) + 1e-6
    return (intersection / union).mean().item()


def main():
    parser = argparse.ArgumentParser(description="3D CNN Voxel Classification & Reconstruction")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "modelnet10"])
    parser.add_argument("--categories", type=str, default="chair,bed,sofa", help="ModelNet10 categories")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=24)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quick-test", action="store_true", help="Run fast smoke test")
    args = parser.parse_args()

    ensure_dirs()
    device = get_device(args.device)
    epochs = 2 if args.quick_test else args.epochs

    if args.dataset == "modelnet10":
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]
        max_train = 40 if args.quick_test else args.max_train_samples
        max_val = 16 if args.quick_test else args.max_val_samples

        train_ds = ModelNet10VoxelDataset(
            split="train",
            resolution=args.resolution,
            categories=cats,
            max_samples=max_train,
        )
        val_ds = ModelNet10VoxelDataset(
            split="test",
            resolution=args.resolution,
            categories=cats,
            max_samples=max_val,
        )
        class_names = train_ds.categories
    else:
        num_samples = 40 if args.quick_test else 320
        train_ds = SyntheticVoxelDataset(num_samples=num_samples, resolution=args.resolution, seed=42)
        val_ds = SyntheticVoxelDataset(num_samples=max(16, num_samples // 4), resolution=args.resolution, seed=99)
        class_names = SHAPE_NAMES

    print(
        f"[02_voxel_3d_cnn] Dataset: {args.dataset.upper()} ({len(class_names)} classes) "
        f"| Resolution: {args.resolution}^3 | Device: {device} | Epochs: {epochs}"
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    clf = VoxelClassifier3D(num_classes=len(class_names)).to(device)
    recon = SingleViewVoxelReconstructor(latent_dim=64).to(device)

    opt_clf = optim.Adam(clf.parameters(), lr=args.lr)
    opt_recon = optim.Adam(recon.parameters(), lr=args.lr)
    ce_loss_fn = nn.CrossEntropyLoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        clf.train()
        recon.train()
        total_clf_loss, total_recon_loss = 0.0, 0.0
        for proj2d, vox3d, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            proj2d, vox3d, labels = proj2d.to(device), vox3d.to(device), labels.to(device)

            # 1. 3D Voxel Classification step
            opt_clf.zero_grad()
            logits_cls = clf(vox3d)
            loss_clf = ce_loss_fn(logits_cls, labels)
            loss_clf.backward()
            opt_clf.step()

            # 2. 2D-to-3D Voxel Reconstruction step
            opt_recon.zero_grad()
            logits_vox = recon(proj2d)
            loss_recon = bce_loss_fn(logits_vox, vox3d)
            loss_recon.backward()
            opt_recon.step()

            total_clf_loss += loss_clf.item()
            total_recon_loss += loss_recon.item()

        # Validation
        clf.eval()
        recon.eval()
        correct, total, iou_sum, batches = 0, 0, 0.0, 0
        with torch.no_grad():
            for proj2d, vox3d, labels in val_loader:
                proj2d, vox3d, labels = proj2d.to(device), vox3d.to(device), labels.to(device)
                preds = clf(vox3d).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                iou_sum += compute_voxel_iou(recon(proj2d), vox3d)
                batches += 1

        val_acc = 100 * correct / max(1, total)
        mean_iou = iou_sum / max(1, batches)
        print(
            f"Epoch {epoch:02d} | Clf Loss: {total_clf_loss/len(train_loader):.4f} "
            f"| Val Acc: {val_acc:.1f}% | Recon Loss: {total_recon_loss/len(train_loader):.4f} "
            f"| Val 3D IoU: {mean_iou:.3f}"
        )

    # Save checkpoints
    ckpt_suffix = f"_{args.dataset}" if args.dataset != "synthetic" else ""
    torch.save(clf.state_dict(), os.path.join(DEFAULT_CHECKPOINT_DIR, f"02_voxel_clf{ckpt_suffix}.pt"))
    torch.save(recon.state_dict(), os.path.join(DEFAULT_CHECKPOINT_DIR, f"02_voxel_recon{ckpt_suffix}.pt"))

    # Visualize examples: 2D Input Projection | Ground Truth 3D | Predicted 3D
    clf.eval()
    recon.eval()
    num_vis = min(4, len(val_ds))
    fig = plt.figure(figsize=(12, 2.5 * num_vis))
    with torch.no_grad():
        for i in range(num_vis):
            proj2d, vox3d, label = val_ds[i]
            pred_vox = torch.sigmoid(recon(proj2d.unsqueeze(0).to(device))).squeeze().cpu().numpy()
            pred_cls = clf(vox3d.unsqueeze(0).to(device)).argmax(dim=1).item()
            cls_name = class_names[label] if label < len(class_names) else f"cls_{label}"
            pred_name = class_names[pred_cls] if pred_cls < len(class_names) else f"cls_{pred_cls}"

            ax1 = fig.add_subplot(num_vis, 3, i * 3 + 1)
            ax1.imshow(proj2d.squeeze().numpy(), cmap="viridis")
            ax1.set_title(f"2D Input ({cls_name})", fontsize=10)
            ax1.axis("off")

            ax2 = fig.add_subplot(num_vis, 3, i * 3 + 2, projection="3d")
            plot_voxels_on_ax(ax2, vox3d.squeeze().numpy(), title=f"GT 3D ({cls_name})", color="steelblue")

            ax3 = fig.add_subplot(num_vis, 3, i * 3 + 3, projection="3d")
            plot_voxels_on_ax(
                ax3,
                pred_vox,
                title=f"Pred 3D (Clf: {pred_name})",
                color="coral",
            )

    out_name = f"02_voxel_3d_cnn_{args.dataset}.png" if args.dataset != "synthetic" else "02_voxel_3d_cnn.png"
    out_path = os.path.join(DEFAULT_SAMPLES_DIR, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
