"""
01_multiview_cnn.py
===================
Multi-View Convolutional Neural Network (MVCNN) for 3D Shape Classification.
Based on Su et al. (ICCV 2015).

Key Concepts:
- Bridging 2D CNNs to 3D understanding via multi-view rendered images.
- Shared CNN1: Extracts per-view 2D feature representations.
- View Pooling: Symmetric element-wise max-pooling across all V camera views.
- CNN2: Predicts 3D shape category from the view-pooled shape descriptor.
- Robustness: Multi-view aggregation eliminates single-view perspective ambiguity.
- Dual Dataset Support:
    1. Synthetic Analytical 3D Primitives (Zero-download, instant smoke tests).
    2. Princeton ModelNet10 Real CAD Meshes (Auto-downloaded & cached for fast MPS training).
"""

import argparse
import os
import math
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_SAMPLES_DIR,
    MODELNET10_CATEGORIES,
    SHAPE_NAMES,
    download_and_extract_modelnet10,
    ensure_dirs,
    evaluate_shape_sdf,
    get_device,
    load_off_mesh,
    plot_pointcloud_on_ax,
    render_mesh_multiview,
    sample_surface_points,
)


def render_multiview_silhouettes(
    shape_name: str,
    num_views: int = 12,
    img_size: int = 32,
    cam_dist: float = 2.2,
    elevation_deg: float = 25.0,
) -> np.ndarray:
    """
    Renders V multi-view depth/silhouette images around an analytical 3D shape
    by sphere-tracing camera rays against evaluate_shape_sdf.
    Returns:
        views: (num_views, 1, img_size, img_size) float32 in [0, 1]
    """
    H = W = img_size
    elev = math.radians(elevation_deg)
    y_cam = cam_dist * math.sin(elev)
    r_cam = cam_dist * math.cos(elev)

    # Pixel grid in normalized camera image plane [-0.8, 0.8]
    px = np.linspace(-0.8, 0.8, W, dtype=np.float32)
    py = np.linspace(0.8, -0.8, H, dtype=np.float32)
    gx, gy = np.meshgrid(px, py, indexing="xy")
    ray_dirs_local = np.stack([gx, gy, -np.ones_like(gx)], axis=-1)
    ray_dirs_local /= np.linalg.norm(ray_dirs_local, axis=-1, keepdims=True)

    views = []
    for v in range(num_views):
        az = 2.0 * math.pi * v / num_views
        cam_pos = np.array([r_cam * math.cos(az), y_cam, r_cam * math.sin(az)], dtype=np.float32)

        forward = -cam_pos / np.linalg.norm(cam_pos)
        up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up_world)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)
        R_c2w = np.stack([right, up, -forward], axis=-1)

        ray_dirs = ray_dirs_local @ R_c2w.T
        ray_dirs = ray_dirs.reshape(-1, 3)
        ray_origins = np.broadcast_to(cam_pos, ray_dirs.shape)

        t = np.full(len(ray_dirs), 1.0, dtype=np.float32)
        for _ in range(5):
            pts = ray_origins + t[:, None] * ray_dirs
            sdf = evaluate_shape_sdf(shape_name, pts)
            t += np.maximum(sdf, 0.02)

        final_pts = ray_origins + t[:, None] * ray_dirs
        hit = evaluate_shape_sdf(shape_name, final_pts) < 0.06
        depth_img = np.zeros(len(ray_dirs), dtype=np.float32)
        depth_img[hit] = np.clip(1.3 - (t[hit] - 1.2) / 1.5, 0.2, 1.0)
        views.append(depth_img.reshape(1, H, W))

    return np.array(views, dtype=np.float32)


class SyntheticMultiViewDataset(Dataset):
    """Dataset yielding (views: (V, 1, H, W), label: int) for analytical primitives."""

    def __init__(
        self,
        num_samples: int = 240,
        num_views: int = 12,
        img_size: int = 32,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.views = []
        self.labels = []
        self.classes = SHAPE_NAMES
        self.files = [None] * num_samples

        base_views = {
            name: render_multiview_silhouettes(name, num_views=num_views, img_size=img_size)
            for name in SHAPE_NAMES
        }

        for i in range(num_samples):
            cls_idx = int(rng.integers(0, len(SHAPE_NAMES)))
            name = SHAPE_NAMES[cls_idx]
            v_imgs = base_views[name].copy()

            # Random circular viewpoint shift (rotation invariance)
            shift = int(rng.integers(0, num_views))
            v_imgs = np.roll(v_imgs, shift=shift, axis=0)

            # Slight noise
            noise = rng.normal(0.0, 0.02, size=v_imgs.shape).astype(np.float32)
            v_imgs = np.clip(v_imgs + noise, 0.0, 1.0)

            self.views.append(v_imgs)
            self.labels.append(cls_idx)

        self.views = torch.from_numpy(np.array(self.views, dtype=np.float32))
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.views[idx], self.labels[idx]


class ModelNet10MultiViewDataset(Dataset):
    """
    ModelNet10 CAD Multi-View Dataset (Su et al., ICCV 2015).
    Loads 3D meshes (.off), renders V camera orbit projections, and caches
    pre-rendered tensors to disk for high-speed training on MPS / CUDA / CPU.
    """

    def __init__(
        self,
        split: str = "train",
        categories: Optional[List[str]] = None,
        num_views: int = 12,
        img_size: int = 64,
        max_samples: Optional[int] = None,
        data_dir: Optional[str] = None,
        seed: int = 42,
    ):
        modelnet_dir = download_and_extract_modelnet10(data_dir)
        if categories is None or "all" in categories or not categories:
            self.classes = MODELNET10_CATEGORIES
        else:
            self.classes = [
                c.strip().lower()
                for c in categories
                if c.strip().lower() in MODELNET10_CATEGORIES
            ]
            if not self.classes:
                self.classes = MODELNET10_CATEGORIES

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_views = num_views
        self.img_size = img_size
        self.split = split

        cls_tag = (
            "_".join(sorted(self.classes))
            if len(self.classes) <= 3
            else f"{len(self.classes)}classes"
        )
        cache_name = f"modelnet10_{split}_{cls_tag}_{num_views}v_{img_size}x{img_size}.pt"
        cache_dir = os.path.join(modelnet_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, cache_name)

        if os.path.exists(cache_file):
            print(f"[ModelNet10] Loading cached multi-view dataset from {cache_file}...")
            cached = torch.load(cache_file, weights_only=False)
            self.views = cached["views"]
            self.labels = cached["labels"]
            self.files = cached.get("files", [])
        else:
            print(
                f"[ModelNet10] Pre-rendering {num_views} views at {img_size}x{img_size} "
                f"for split '{split}' ({len(self.classes)} classes)..."
            )
            off_files = []
            file_labels = []
            for c in self.classes:
                c_dir = os.path.join(modelnet_dir, c, split)
                if not os.path.exists(c_dir):
                    continue
                for fname in sorted(os.listdir(c_dir)):
                    if fname.endswith(".off"):
                        off_files.append(os.path.join(c_dir, fname))
                        file_labels.append(self.class_to_idx[c])

            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(off_files))
            off_files = [off_files[p] for p in perm]
            file_labels = [file_labels[p] for p in perm]

            rendered_views = []
            valid_labels = []
            valid_files = []
            for fpath, lab in tqdm(
                zip(off_files, file_labels), total=len(off_files), desc=f"Rendering {split}"
            ):
                try:
                    verts, faces = load_off_mesh(fpath)
                    v_imgs = render_mesh_multiview(
                        verts, faces, num_views=num_views, img_size=img_size
                    )
                    rendered_views.append(v_imgs)
                    valid_labels.append(lab)
                    valid_files.append(fpath)
                except Exception:
                    continue

            self.views = torch.from_numpy(np.array(rendered_views, dtype=np.float32))
            self.labels = torch.tensor(valid_labels, dtype=torch.long)
            self.files = valid_files

            print(f"[ModelNet10] Caching {len(self.views)} shapes to {cache_file}...")
            torch.save(
                {"views": self.views, "labels": self.labels, "files": self.files},
                cache_file,
            )

        if max_samples is not None and max_samples < len(self.labels):
            self.views = self.views[:max_samples]
            self.labels = self.labels[:max_samples]
            self.files = self.files[:max_samples]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.views[idx], self.labels[idx]


class MultiViewCNN(nn.Module):
    """
    Multi-View CNN Architecture (Su et al., ICCV 2015).
    - CNN1: Shared 2D ConvNet extracting descriptors per view.
    - View Pooling: Element-wise max-pooling across view axis.
    - CNN2: Fully connected classifier on pooled shape descriptor.
    """

    def __init__(self, num_classes: int = 4, feature_dim: int = 128):
        super().__init__()
        # CNN1: Shared per-view feature extractor
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(128 * 4, feature_dim),
            nn.ReLU(inplace=True),
        )

        # CNN2: Shape descriptor classifier
        self.cnn2 = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor, view_pooling: bool = True) -> torch.Tensor:
        """
        Args:
            x: (B, V, 1, H, W)
            view_pooling: whether to perform element-wise view pooling.
        """
        B, V, C, H, W = x.shape
        views = x.view(B * V, C, H, W)
        view_feats = self.cnn1(views)  # (B * V, feature_dim)

        if view_pooling and V > 1:
            view_feats = view_feats.view(B, V, -1)
            shape_feat = view_feats.max(dim=1)[0]  # Element-wise View Pooling (B, feature_dim)
        else:
            shape_feat = view_feats.view(B, V, -1).mean(dim=1)

        logits = self.cnn2(shape_feat)
        return logits


def main():
    parser = argparse.ArgumentParser(description="Multi-View CNN (MVCNN) 3D Shape Classification")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "modelnet10"],
        help="Dataset to use: 'synthetic' (zero-download) or 'modelnet10' (real CAD dataset)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="all",
        help="Comma-separated subset of ModelNet10 classes (e.g. 'chair,bed,sofa') or 'all'",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-views", type=int, default=12)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = get_device(args.device)
    epochs = 2 if args.quick_test else args.epochs
    num_views = min(args.num_views, 12)

    if args.dataset == "modelnet10":
        cats = [c.strip().lower() for c in args.categories.split(",")] if args.categories != "all" else None
        img_size = args.img_size if args.img_size is not None else 64
        max_train = 60 if args.quick_test else args.max_train_samples
        max_val = 20 if args.quick_test else args.max_val_samples

        train_ds = ModelNet10MultiViewDataset(
            split="train",
            categories=cats,
            num_views=num_views,
            img_size=img_size,
            max_samples=max_train,
            data_dir=DEFAULT_DATA_DIR,
        )
        val_ds = ModelNet10MultiViewDataset(
            split="test",
            categories=cats,
            num_views=num_views,
            img_size=img_size,
            max_samples=max_val,
            data_dir=DEFAULT_DATA_DIR,
        )
        class_names = train_ds.classes
    else:
        img_size = args.img_size if args.img_size is not None else 32
        num_samples = 40 if args.quick_test else 240
        train_ds = SyntheticMultiViewDataset(
            num_samples=num_samples, num_views=num_views, img_size=img_size, seed=42
        )
        val_ds = SyntheticMultiViewDataset(
            num_samples=max(16, num_samples // 4), num_views=num_views, img_size=img_size, seed=99
        )
        class_names = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(
        f"[01_multiview_cnn] Dataset: {args.dataset.upper()} ({len(class_names)} classes) "
        f"| Views: {num_views} | Img: {img_size}x{img_size} | Device: {device} | Epochs: {epochs}"
    )

    model = MultiViewCNN(num_classes=len(class_names), feature_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for views, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            views, labels = views.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(views, view_pooling=True)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        mv_correct, sv_correct, total = 0, 0, 0
        with torch.no_grad():
            for views, labels in val_loader:
                views, labels = views.to(device), labels.to(device)
                total += labels.size(0)

                # Multi-view classification with view-pooling
                logits_mv = model(views, view_pooling=True)
                mv_correct += (logits_mv.argmax(dim=1) == labels).sum().item()

                # Single-view ablation: evaluate only the first camera view
                logits_sv = model(views[:, :1], view_pooling=False)
                sv_correct += (logits_sv.argmax(dim=1) == labels).sum().item()

        print(
            f"Epoch {epoch:02d} | Train Loss: {total_loss/len(train_loader):.4f} "
            f"| Multi-View Val Acc: {100*mv_correct/total:.1f}% "
            f"| Single-View Val Acc: {100*sv_correct/total:.1f}%"
        )

    ckpt_name = f"01_multiview_cnn_{args.dataset}.pt"
    torch.save(model.state_dict(), os.path.join(DEFAULT_CHECKPOINT_DIR, ckpt_name))

    # Visualization: 3D CAD Shape with Virtual Camera Orbit + All Rendered Views
    model.eval()
    sample_idx = 0
    views_t, label_gt = val_ds[sample_idx]
    with torch.no_grad():
        pred_logits = model(views_t.unsqueeze(0).to(device), view_pooling=True)
        pred_cls = pred_logits.argmax(dim=1).item()
        probs = torch.softmax(pred_logits, dim=1).squeeze().cpu().numpy()

    fig = plt.figure(figsize=(15, 6))

    # Panel 1: 3D Shape + Virtual Camera Orbit Ring
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    if args.dataset == "modelnet10" and val_ds.files[sample_idx] is not None:
        verts_cad, _ = load_off_mesh(val_ds.files[sample_idx])
        plot_pointcloud_on_ax(
            ax3d, verts_cad, title=f"ModelNet10 ({class_names[label_gt]})", s=3.0
        )
    else:
        pts_surface = sample_surface_points(class_names[label_gt], num_points=800)
        plot_pointcloud_on_ax(
            ax3d, pts_surface, title=f"3D Shape ({class_names[label_gt]})", s=2.5
        )

    # Draw camera orbit ring
    cam_r = 1.3
    cam_y = 0.55
    az_vals = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    c_x = cam_r * np.cos(az_vals)
    c_z = cam_r * np.sin(az_vals)
    ax3d.scatter(
        c_x, c_z, np.full_like(c_x, cam_y), color="crimson", s=40, marker="^", label="Cameras"
    )
    ax3d.plot(c_x, c_z, np.full_like(c_x, cam_y), color="red", linestyle="--", alpha=0.5)
    ax3d.legend(loc="upper right", fontsize=8)

    # Panel 2: Rendered Views Grid
    ax_grid = fig.add_subplot(1, 3, 2)
    rows = 3
    cols = (num_views + rows - 1) // rows
    v_np = views_t.squeeze(1).numpy()
    tile_h = v_np.shape[1]
    canvas = np.zeros((rows * tile_h, cols * tile_h), dtype=np.float32)
    for idx in range(num_views):
        r = idx // cols
        c = idx % cols
        canvas[r * tile_h : (r + 1) * tile_h, c * tile_h : (c + 1) * tile_h] = v_np[idx]
    ax_grid.imshow(canvas, cmap="bone")
    ax_grid.set_title(f"Rendered {num_views} Views ({args.dataset})", fontsize=10)
    ax_grid.axis("off")

    # Panel 3: Classification Probabilities
    ax_bar = fig.add_subplot(1, 3, 3)
    bars = ax_bar.barh(class_names, probs, color="teal")
    if pred_cls == label_gt:
        bars[pred_cls].set_color("forestgreen")
    else:
        bars[pred_cls].set_color("crimson")
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xlabel("Probability")
    ax_bar.set_title(
        f"MVCNN: {class_names[pred_cls]} (GT: {class_names[label_gt]})",
        fontsize=10,
    )

    out_name = f"01_multiview_cnn_{args.dataset}.png" if args.dataset != "synthetic" else "01_multiview_cnn.png"
    out_path = os.path.join(DEFAULT_SAMPLES_DIR, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
