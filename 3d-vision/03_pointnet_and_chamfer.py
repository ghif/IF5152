"""
03_pointnet_and_chamfer.py
==========================
Point Cloud Deep Learning with PointNet, DGCNN, and Chamfer Distance.

Key Concepts:
- Unordered 3D Point Clouds (N x 3) and Permutation Invariance.
- PointNet Architecture: Shared MLPs (1D Convolutions) + T-Net + Symmetric Max Pooling.
- Vectorized Chamfer Distance (CD) in PyTorch for set-to-set supervision.
- Point Cloud Autoencoder reconstructing 3D shapes from a 64-D bottleneck.
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
    get_device,
    load_off_mesh,
    plot_pointcloud_on_ax,
    sample_mesh_surface_points,
    sample_surface_points,
)


class ModelNet10PointCloudDataset(Dataset):
    """
    ModelNet10 CAD dataset uniformly sampled as N x 3 surface point clouds.
    Pre-processes and caches .pt tensors into data/ModelNet10/cache/ for instant loading.
    """

    def __init__(
        self,
        split: str = "train",
        num_points: int = 512,
        categories: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        data_dir: Optional[str] = None,
    ):
        self.num_points = num_points
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
            f"modelnet10_pts_{split}_{cat_key}_pts{num_points}{limit_str}.pt",
        )

        if os.path.exists(cache_file):
            print(f"[ModelNet10] Loading cached point cloud dataset from {cache_file}...")
            cached = torch.load(cache_file, weights_only=False)
            self.clouds = cached["clouds"]
            self.labels = cached["labels"]
            return

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

        print(f"[ModelNet10] Sampling {num_points} points for {len(file_list)} {split} meshes...")
        cloud_list, lbl_list = [], []
        rng = np.random.default_rng(42)

        for filepath, lbl in tqdm(file_list, desc=f"Sampling {split}", leave=False):
            try:
                verts, faces = load_off_mesh(filepath)
                pts = sample_mesh_surface_points(verts, faces, num_points=num_points, rng=rng)
                cloud_list.append(pts)
                lbl_list.append(lbl)
            except Exception as e:
                print(f"[Warning] Failed to sample {filepath}: {e}")

        self.clouds = torch.from_numpy(np.array(cloud_list, dtype=np.float32))
        self.labels = torch.tensor(lbl_list, dtype=torch.long)

        print(f"[ModelNet10] Caching {len(self.labels)} point clouds to {cache_file}...")
        torch.save(
            {
                "clouds": self.clouds,
                "labels": self.labels,
            },
            cache_file,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.clouds[idx], self.labels[idx]


def chamfer_distance(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """
    Vectorized bidirectional Chamfer Distance between two point cloud batches.
    Args:
        p1: (B, N, 3)
        p2: (B, M, 3)
    Returns:
        Scalar mean Chamfer Distance across the batch.
    """
    # Pairwise squared Euclidean distance: (B, N, M)
    dist = torch.cdist(p1, p2, p=2.0) ** 2
    min_p1_to_p2 = dist.min(dim=2)[0].mean(dim=1)  # (B,)
    min_p2_to_p1 = dist.min(dim=1)[0].mean(dim=1)  # (B,)
    return (min_p1_to_p2 + min_p2_to_p1).mean()


class SyntheticPointCloudDataset(Dataset):
    """Generates N-point surface clouds with random SO(3) rotations."""

    def __init__(self, num_samples: int = 200, num_points: int = 512, seed: int = 42):
        rng = np.random.default_rng(seed)
        clouds = []
        labels = []

        # Pre-sample canonical surfaces once per class for fast batch generation
        canonical = {
            name: sample_surface_points(name, num_points=num_points * 2, rng=rng)
            for name in SHAPE_NAMES
        }

        for _ in range(num_samples):
            cls_idx = int(rng.integers(0, len(SHAPE_NAMES)))
            name = SHAPE_NAMES[cls_idx]
            idx = rng.choice(len(canonical[name]), size=num_points, replace=False)
            pts = canonical[name][idx].copy()

            # Random Y-axis rotation + slight jitter
            theta = rng.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            pts = pts @ rot.T + rng.normal(0, 0.015, size=pts.shape).astype(np.float32)

            clouds.append(pts)
            labels.append(cls_idx)

        self.clouds = torch.from_numpy(np.array(clouds, dtype=np.float32))
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.clouds[idx], self.labels[idx]


class TNet(nn.Module):
    """Spatial Transformer Network predicting a 3x3 alignment matrix."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv1d(k, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, k * k),
        )
        # Initialize bias to identity matrix
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)
        self.register_buffer("eye", torch.eye(k).view(1, k * k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, k, N)
        h = self.mlp(x).max(dim=2)[0]
        mat = self.fc(h) + self.eye
        return mat.view(-1, self.k, self.k)


class PointNetEncoder(nn.Module):
    """Permutation-invariant PointNet encoder: (B, N, 3) -> (B, latent_dim)."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.tnet = TNet(k=3)
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, latent_dim, 1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        # pts: (B, N, 3) -> (B, 3, N)
        x = pts.transpose(1, 2)
        trans = self.tnet(x)
        x = torch.bmm(trans, x)
        feat = self.mlp(x)
        # Symmetric max pooling across all N points
        global_feat = feat.max(dim=2)[0]
        return global_feat


class EdgeConv(nn.Module):
    """
    Dynamic Graph CNN (DGCNN) EdgeConv layer (Wang et al., ToG 2019).
    Constructs a k-NN graph dynamically in feature space and applies
    channel-wise max pooling over edge features [x_i, x_j - x_i].
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 16):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N)
        B, C, N = x.shape
        x_t = x.transpose(1, 2)  # (B, N, C)
        dists = torch.cdist(x_t, x_t, p=2.0)  # (B, N, N)
        knn_idx = dists.topk(k=min(self.k, N), dim=-1, largest=False)[1]  # (B, N, k)

        idx_expanded = knn_idx.unsqueeze(1).expand(B, C, N, -1)  # (B, C, N, k)
        x_expanded = x.unsqueeze(-1).expand(B, C, N, knn_idx.shape[-1])
        neighbors = torch.gather(x.unsqueeze(2).expand(B, C, N, N), dim=3, index=idx_expanded)

        edge_feat = torch.cat([x_expanded, neighbors - x_expanded], dim=1)  # (B, 2C, N, k)
        out = self.conv(edge_feat).max(dim=-1)[0]  # (B, out_channels, N)
        return out


class DGCNNEncoder(nn.Module):
    """Dynamic Graph CNN encoder stacking EdgeConv blocks + global max pooling."""

    def __init__(self, latent_dim: int = 128, k: int = 16):
        super().__init__()
        self.ec1 = EdgeConv(3, 64, k=k)
        self.ec2 = EdgeConv(64, 64, k=k)
        self.ec3 = EdgeConv(64, latent_dim, k=k)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        x = pts.transpose(1, 2)  # (B, 3, N)
        x1 = self.ec1(x)
        x2 = self.ec2(x1)
        x3 = self.ec3(x2)
        return x3.max(dim=2)[0]


class PointNetAutoencoderAndClassifier(nn.Module):
    """Dual-head Point Cloud network (PointNet or DGCNN): classifies & reconstructs N x 3 points."""

    def __init__(
        self,
        num_points: int = 512,
        num_classes: int = 4,
        latent_dim: int = 128,
        arch: str = "pointnet",
    ):
        super().__init__()
        self.num_points = num_points
        if arch == "dgcnn":
            self.encoder = DGCNNEncoder(latent_dim=latent_dim)
        else:
            self.encoder = PointNetEncoder(latent_dim=latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_points * 3),
        )

    def forward(self, pts: torch.Tensor):
        z = self.encoder(pts)
        logits = self.classifier(z)
        recon = self.decoder(z).view(-1, self.num_points, 3)
        return logits, recon


def main():
    parser = argparse.ArgumentParser(description="PointNet / DGCNN Classification & Chamfer Autoencoder")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "modelnet10"])
    parser.add_argument("--categories", type=str, default="chair,bed,sofa", help="ModelNet10 categories")
    parser.add_argument("--arch", type=str, default="pointnet", choices=["pointnet", "dgcnn"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = get_device(args.device)
    epochs = 2 if args.quick_test else args.epochs

    if args.dataset == "modelnet10":
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]
        max_train = 40 if args.quick_test else args.max_train_samples
        max_val = 16 if args.quick_test else args.max_val_samples

        train_ds = ModelNet10PointCloudDataset(
            split="train",
            num_points=args.num_points,
            categories=cats,
            max_samples=max_train,
        )
        val_ds = ModelNet10PointCloudDataset(
            split="test",
            num_points=args.num_points,
            categories=cats,
            max_samples=max_val,
        )
        class_names = train_ds.categories
    else:
        num_samples = 40 if args.quick_test else 320
        train_ds = SyntheticPointCloudDataset(num_samples=num_samples, num_points=args.num_points, seed=42)
        val_ds = SyntheticPointCloudDataset(num_samples=max(16, num_samples // 4), num_points=args.num_points, seed=99)
        class_names = SHAPE_NAMES

    print(
        f"[03_pointnet_and_chamfer] Dataset: {args.dataset.upper()} ({len(class_names)} classes) "
        f"| Arch: {args.arch} | Device: {device} | Points: {args.num_points} | Epochs: {epochs}"
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = PointNetAutoencoderAndClassifier(
        num_points=args.num_points, num_classes=len(class_names), arch=args.arch
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ce_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for pts, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            pts, labels = pts.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, recon = model(pts)
            loss_ce = ce_loss_fn(logits, labels)
            loss_cd = chamfer_distance(recon, pts)
            loss = loss_ce + 10.0 * loss_cd
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total, val_cd = 0, 0, 0.0
        with torch.no_grad():
            for pts, labels in val_loader:
                pts, labels = pts.to(device), labels.to(device)
                logits, recon = model(pts)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
                val_cd += chamfer_distance(recon, pts).item()

        val_acc = 100 * correct / max(1, total)
        mean_cd = val_cd / max(1, len(val_loader))
        print(
            f"Epoch {epoch:02d} | Train Loss: {total_loss/len(train_loader):.4f} "
            f"| Val Acc: {val_acc:.1f}% | Val Chamfer: {mean_cd:.5f}"
        )

    ckpt_suffix = f"_{args.dataset}_{args.arch}" if args.dataset != "synthetic" else f"_{args.arch}"
    torch.save(model.state_dict(), os.path.join(DEFAULT_CHECKPOINT_DIR, f"03_pointnet{ckpt_suffix}.pt"))

    # Visualize Ground Truth vs Reconstructed Point Clouds
    model.eval()
    num_vis = min(4, len(val_ds))
    fig = plt.figure(figsize=(10, 3 * num_vis))
    with torch.no_grad():
        for i in range(num_vis):
            pts, label = val_ds[i]
            logits, recon = model(pts.unsqueeze(0).to(device))
            pred_cls = logits.argmax(dim=1).item()
            recon_np = recon.squeeze(0).cpu().numpy()
            true_name = class_names[label] if label < len(class_names) else f"cls_{label}"
            pred_name = class_names[pred_cls] if pred_cls < len(class_names) else f"cls_{pred_cls}"

            ax1 = fig.add_subplot(num_vis, 2, i * 2 + 1, projection="3d")
            plot_pointcloud_on_ax(ax1, pts.numpy(), title=f"Input GT ({true_name})")

            ax2 = fig.add_subplot(num_vis, 2, i * 2 + 2, projection="3d")
            plot_pointcloud_on_ax(
                ax2,
                recon_np,
                title=f"Chamfer Recon (Pred: {pred_name})",
            )

    out_name = f"03_pointnet_chamfer_{args.dataset}.png" if args.dataset != "synthetic" else "03_pointnet_chamfer.png"
    out_path = os.path.join(DEFAULT_SAMPLES_DIR, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
