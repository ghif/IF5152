"""
========================================================================================
IF5152: Video Understanding - Session 3 Lab
File: 04_train_late_fusion_head.py
========================================================================================

PEDAGOGICAL & THEORETICAL FOUNDATIONS:
----------------------------------------------------------------------------------------
1. Lecture Slides 18-20 & 63: "Training Temporal Sequence Models on Video"
   - Video models require modeling transitions across time.
   - End-to-end training of 3D CNNs or full video transformers requires massive multi-GPU
     clusters and days of compute.
   - Industrial Standard Practice: Freeze an ImageNet-pretrained 2D spatial backbone
     (e.g., MobileNetV3 / ResNet) and train a lightweight temporal head (GRU, LSTM, or
     Temporal Attention) on top of the extracted frame representations!

2. The Feature Caching Speedup (MPS Optimization):
   - In production pipelines, video decoding (PyAV / FFmpeg) on CPU is the primary bottleneck.
   - If we re-decode every video on every epoch, 90% of training time is wasted waiting for CPU.
   - By pre-extracting spatial features ONCE on Apple Silicon GPU (MPS) and caching them in
     unified memory, each training epoch drops from minutes to FRACTIONS OF A SECOND!

3. Inductive Bias Comparison:
   - Temporal Mean Pooling (Slide 19): Order-invariant baseline.
   - Recurrent GRU (Slide 63): Order-aware recurrent modeling with Backpropagation
     Through Time (BPTT).

DATASET MODES:
  1. 'ucf_subset' (Default): Automatically downloads & extracts a compact 5-class subset
     of UCF-101 (ApplyEyeMakeup, BaseballPitch, Basketball, BenchPress, BandMarching).
  2. 'synthetic': Instantly generates 5 synthetic motion classes (drift, bounce, rotate, pulse)
     in ~1 second for 100% offline verification with zero bandwidth requirements.

USAGE:
  # Fast training with GRU on Apple Silicon M4 MPS:
  python 04_train_late_fusion_head.py --device mps --epochs 20 --fusion gru

  # Baseline comparison with Temporal Mean Pooling:
  python 04_train_late_fusion_head.py --device mps --epochs 20 --fusion mean

  # 100% Offline test using synthetic motion videos:
  python 04_train_late_fusion_head.py --dataset synthetic --device mps
========================================================================================
"""
import argparse
import glob
import math
import os
import random
import tarfile
import time
import urllib.request
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from utils import get_device, print_device_info, read_video_pyav, resolve_data_path


# ======================================================================================
# PART 1: DATASET SETUP (UCF-101 MINI & SYNTHETIC MOTION GENERATOR)
# ======================================================================================

DEFAULT_UCF_CLASSES = [
    "ApplyEyeMakeup",
    "BaseballPitch",
    "Basketball",
    "BenchPress",
    "BandMarching"
]

UCF_SUBSET_URL = (
    "https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/main/UCF101_subset.tar.gz"
)


def setup_ucf_subset(target_dir: str, target_classes: List[str], max_clips_per_class: int = 15) -> str:
    """
    Streams and extracts a targeted subset of UCF-101 videos directly from Hugging Face Hub.
    Downloads only the specified classes and clip count to minimize disk space and bandwidth.
    """
    os.makedirs(target_dir, exist_ok=True)
    existing_clips = {cls: len(glob.glob(os.path.join(target_dir, cls, "*.*"))) for cls in target_classes}
    
    if all(count >= max_clips_per_class for count in existing_clips.values()):
        print(f"  [OK] Dataset already present in '{target_dir}': {existing_clips}")
        return target_dir

    print(f"  [DOWNLOAD] Fetching {len(target_classes)} classes ({max_clips_per_class} clips/class) from UCF101-subset...")
    print(f"  Target URL: {UCF_SUBSET_URL}")

    req = urllib.request.Request(UCF_SUBSET_URL, headers={"User-Agent": "Mozilla/5.0"})
    extracted = {cls: existing_clips.get(cls, 0) for cls in target_classes}

    try:
        with urllib.request.urlopen(req) as resp:
            with tarfile.open(fileobj=resp, mode="r|") as tar:
                for member in tar:
                    parts = member.name.split("/")
                    # Archive layout: UCF101_subset/train/ClassName/video.avi
                    if len(parts) >= 4 and parts[2] in target_classes and member.isfile():
                        cls = parts[2]
                        if extracted[cls] < max_clips_per_class:
                            extracted[cls] += 1
                            out_path = os.path.join(target_dir, cls, parts[-1])
                            os.makedirs(os.path.dirname(out_path), exist_ok=True)
                            with open(out_path, "wb") as f:
                                f.write(tar.extractfile(member).read())
                    if all(v >= max_clips_per_class for v in extracted.values()):
                        break
        print(f"  [DOWNLOAD OK] Extracted clips: {extracted}")
    except Exception as e:
        print(f"  [WARNING] Online download failed ({e}). Falling back to synthetic motion dataset.")
        return generate_synthetic_dataset(os.path.join(os.path.dirname(target_dir), "synthetic_actions"))

    return target_dir


def generate_synthetic_dataset(target_dir: str, clips_per_class: int = 12, num_frames: int = 16) -> str:
    """
    Generates synthetic motion videos for 100% offline educational use.
    Classes have distinct temporal patterns:
      1. drift_right: Square moving horizontally left to right
      2. drift_down: Circle moving vertically top to bottom
      3. diagonal_bounce: Triangle bouncing diagonally
      4. rotate_clockwise: Rotating line segment
      5. pulsing_scale: Circle oscillating in radius
    """
    os.makedirs(target_dir, exist_ok=True)
    classes = ["drift_right", "drift_down", "diagonal_bounce", "rotate_clockwise", "pulsing_scale"]

    print(f"  [SYNTHETIC] Generating {len(classes)} classes x {clips_per_class} clips ({num_frames} frames each)...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for cls in classes:
        cls_dir = os.path.join(target_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        if len(glob.glob(os.path.join(cls_dir, "*.mp4"))) >= clips_per_class:
            continue

        for clip_idx in range(clips_per_class):
            filepath = os.path.join(cls_dir, f"synth_{cls}_{clip_idx:02d}.mp4")
            writer = cv2.VideoWriter(filepath, fourcc, 10, (128, 128))
            speed = random.uniform(0.8, 1.2)

            for t in range(num_frames):
                frame = np.zeros((128, 128, 3), dtype=np.uint8)
                # Add background texture/noise
                frame[:] = np.random.randint(15, 25, size=(128, 128, 3), dtype=np.uint8)

                if cls == "drift_right":
                    x = int(10 + (t * 6 * speed)) % 110
                    cv2.rectangle(frame, (x, 50), (x + 20, 70), (0, 200, 255), -1)
                elif cls == "drift_down":
                    y = int(10 + (t * 6 * speed)) % 110
                    cv2.circle(frame, (64, y), 12, (255, 100, 100), -1)
                elif cls == "diagonal_bounce":
                    pos = int(math.sin(t * 0.4 * speed) * 40 + 64)
                    pts = np.array([[pos, pos], [pos + 20, pos], [pos + 10, pos - 20]])
                    cv2.drawContours(frame, [pts], 0, (100, 255, 100), -1)
                elif cls == "rotate_clockwise":
                    angle = (t * 20 * speed) * math.pi / 180.0
                    x2 = int(64 + 35 * math.cos(angle))
                    y2 = int(64 + 35 * math.sin(angle))
                    cv2.line(frame, (64, 64), (x2, y2), (255, 255, 0), 4)
                elif cls == "pulsing_scale":
                    radius = int(8 + abs(math.sin(t * 0.5 * speed)) * 25)
                    cv2.circle(frame, (64, 64), radius, (200, 50, 200), -1)

                writer.write(frame)
            writer.release()

    print(f"  [SYNTHETIC OK] Synthetic dataset ready at '{target_dir}'.")
    return target_dir


# ======================================================================================
# PART 2: DATASET & PRE-COMPUTED FEATURE CACHE
# ======================================================================================

class VideoClipDataset:
    """Manages video file discovery, label mapping, and train/val splitting."""
    def __init__(self, root_dir: str, num_frames: int = 16, split_ratio: float = 0.8, seed: int = 42):
        self.root_dir = root_dir
        self.num_frames = num_frames
        
        # Discover class subfolders
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith(".")
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect video files per class
        random.seed(seed)
        self.train_samples: List[Tuple[str, int]] = []
        self.val_samples: List[Tuple[str, int]] = []

        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            clips = sorted(glob.glob(os.path.join(cls_folder, "*.*")))
            random.shuffle(clips)
            split_idx = max(1, int(len(clips) * split_ratio))
            
            for clip in clips[:split_idx]:
                self.train_samples.append((clip, self.class_to_idx[cls]))
            for clip in clips[split_idx:]:
                self.val_samples.append((clip, self.class_to_idx[cls]))


def cache_spatial_features(
    samples: List[Tuple[str, int]],
    backbone: nn.Module,
    device: torch.device,
    num_frames: int = 16,
    batch_size: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts spatial embeddings across all frames for all videos using the frozen 2D CNN.
    Stores the output tensor [N, T, D] directly in memory on MPS / RAM.
    This eliminates all video decoding overhead during training.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    backbone.eval()
    all_features = []
    all_labels = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for i, (video_path, label) in enumerate(samples, start=1):
            raw_frames = read_video_pyav(video_path, num_frames=num_frames, fast_seek=True)
            pil_frames = [Image.fromarray(f) for f in raw_frames]
            tensors = torch.stack([transform(f) for f in pil_frames]).to(device)  # [T, 3, 224, 224]

            # Forward through 2D CNN backbone
            frame_feats = backbone(tensors)  # [T, D]
            all_features.append(frame_feats.cpu())
            all_labels.append(label)

            if i % 10 == 0 or i == len(samples):
                print(f"    Extracted & cached {i}/{len(samples)} clips...", end="\r", flush=True)

    elapsed = time.perf_counter() - t0
    features_tensor = torch.stack(all_features)  # [N, T, D]
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    print(f"\n    Feature cache constructed: {tuple(features_tensor.shape)} in {elapsed:.2f}s ({elapsed/len(samples)*1000:.1f} ms/clip)")
    return features_tensor, labels_tensor


class CachedFeatureDataset(Dataset):
    """PyTorch Dataset yielding pre-computed spatial feature sequences."""
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ======================================================================================
# PART 3: TEMPORAL HEAD ARCHITECTURES (MEAN VS GRU)
# ======================================================================================

class TemporalHead(nn.Module):
    """
    Trainable temporal sequence model (Lecture Slides 19 & 63):
      - 'mean': Permutation-invariant Average Pooling across frames
      - 'gru' : Bidirectional Recurrent GRU with Backpropagation Through Time (BPTT)
    """
    def __init__(self, in_features: int, num_classes: int, fusion: str = "gru", hidden_dim: int = 128):
        super().__init__()
        self.fusion = fusion

        if fusion == "gru":
            # Bidirectional GRU to capture forward & backward temporal dynamics
            self.gru = nn.GRU(
                input_size=in_features,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True
            )
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        elif fusion == "mean":
            # Slide 19: Direct linear projection from temporally averaged embedding
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [Batch, Time, in_features]
        if self.fusion == "gru":
            gru_out, _ = self.gru(x)              # [Batch, Time, hidden_dim * 2]
            # Pool temporal representations across all time steps
            temporal_repr = gru_out.mean(dim=1)   # [Batch, hidden_dim * 2]
            out = self.dropout(temporal_repr)
            logits = self.classifier(out)         # [Batch, num_classes]
        elif self.fusion == "mean":
            temporal_repr = x.mean(dim=1)         # [Batch, in_features]
            out = self.dropout(temporal_repr)
            logits = self.classifier(out)         # [Batch, num_classes]
        return logits


# ======================================================================================
# PART 4: TRAINING & EVALUATION LOOP
# ======================================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()

        # Optional FP16 autocast on MPS
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    return total_loss / total, (correct / total) * 100.0


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        outputs = model(features)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * targets.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    return total_loss / total, (correct / total) * 100.0, all_preds, all_targets


# ======================================================================================
# MAIN EXECUTION SCRIPT
# ======================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="IF5152 Lecture 11 Lab: Train / Fine-tune Late Fusion Video Head on Apple M4 MPS"
    )
    parser.add_argument("--dataset", choices=["ucf_subset", "synthetic"], default="ucf_subset",
                        help="Dataset source: 'ucf_subset' (downloads UCF101 mini) or 'synthetic' (offline generator)")
    parser.add_argument("--fusion", choices=["gru", "mean"], default="gru",
                        help="Temporal head fusion strategy: 'gru' (recurrent) or 'mean' (pooling)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for temporal training (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for AdamW optimizer (default: 1e-3)")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension for GRU head (default: 128)")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of uniformly sampled frames per clip (default: 16)")
    parser.add_argument("--clips_per_class", type=int, default=12,
                        help="Clips to use per action category (default: 12)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Target device: 'auto' | 'mps' | 'cuda' | 'cpu'")
    args = parser.parse_args()

    # 1. Device Inspection
    device = get_device(args.device)
    print_device_info(device)

    # 2. Setup Dataset
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    print(f"\n--- [STEP 1] Preparing Dataset ({args.dataset.upper()}) ---")

    if args.dataset == "ucf_subset":
        dataset_path = os.path.join(data_root, "ucf101_mini")
        dataset_path = setup_ucf_subset(dataset_path, DEFAULT_UCF_CLASSES, max_clips_per_class=args.clips_per_class)
    else:
        dataset_path = os.path.join(data_root, "synthetic_actions")
        dataset_path = generate_synthetic_dataset(dataset_path, clips_per_class=args.clips_per_class, num_frames=args.num_frames)

    dataset_mgr = VideoClipDataset(dataset_path, num_frames=args.num_frames)
    num_classes = len(dataset_mgr.classes)
    print(f"Dataset Ready: {len(dataset_mgr.train_samples)} Train clips, {len(dataset_mgr.val_samples)} Val clips across {num_classes} classes:")
    for idx, cls_name in enumerate(dataset_mgr.classes):
        print(f"  Class #{idx}: {cls_name}")

    # 3. Load Backbone & Cache Features
    print("\n--- [STEP 2] Precomputing Spatial Features on " + device.type.upper() + " ---")
    print("Loading pretrained MobileNetV3-Small spatial backbone...")
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    backbone = models.mobilenet_v3_small(weights=weights)
    feat_dim = backbone.classifier[0].in_features  # 576
    backbone.classifier = nn.Identity()           # Raw feature extractor
    backbone = backbone.to(device)

    print("Extracting features for training set:")
    train_feats, train_labels = cache_spatial_features(dataset_mgr.train_samples, backbone, device, num_frames=args.num_frames)
    print("Extracting features for validation set:")
    val_feats, val_labels = cache_spatial_features(dataset_mgr.val_samples, backbone, device, num_frames=args.num_frames)

    train_loader = DataLoader(CachedFeatureDataset(train_feats, train_labels), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(CachedFeatureDataset(val_feats, val_labels), batch_size=args.batch_size, shuffle=False)

    # 4. Initialize Trainable Temporal Head
    print(f"\n--- [STEP 3] Initializing Trainable Temporal Head ({args.fusion.upper()}) ---")
    model = TemporalHead(in_features=feat_dim, num_classes=num_classes, fusion=args.fusion, hidden_dim=args.hidden_dim).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters in Temporal Head: {trainable_params:,}")
    print(f"Frozen Backbone Parameters             : {sum(p.numel() for p in backbone.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 5. Training Loop
    print(f"\n--- [STEP 4] Training on {device.type.upper()} ({args.epochs} Epochs) ---")
    print("=" * 68)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Time':<8}")
    print("=" * 68)

    best_val_acc = 0.0
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, f"best_head_{args.fusion}.pt")

    t_train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, preds, targets = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_ms = (time.perf_counter() - t0) * 1000.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "classes": dataset_mgr.classes,
                "fusion": args.fusion,
                "val_acc": val_acc
            }, best_checkpoint_path)
            mark = " ★ (Best)"
        else:
            mark = ""

        print(f"{epoch:02d}/{args.epochs:02d}   {tr_loss:<12.4f} {tr_acc:<11.1f}% {val_loss:<12.4f} {val_acc:<11.1f}% {epoch_ms:5.1f}ms{mark}")

    total_time = time.perf_counter() - t_train_start
    print("=" * 68)
    print(f"Training completed in {total_time:.2f} seconds ({total_time / args.epochs * 1000:.1f} ms/epoch average).")
    print(f"Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"Saved best model checkpoint to: '{best_checkpoint_path}'")

    # 6. Evaluation Summary
    print("\n--- [STEP 5] Final Classification Summary ---")
    _, final_acc, all_preds, all_targets = evaluate(model, val_loader, criterion, device)
    print(f"Overall Validation Accuracy: {final_acc:.2f}%\n")
    print(f"{'Class Name':<25} {'Correct / Total':<18} {'Accuracy'}")
    print("-" * 55)
    for idx, cls_name in enumerate(dataset_mgr.classes):
        cls_mask = [t == idx for t in all_targets]
        cls_total = sum(cls_mask)
        cls_correct = sum(p == t for p, t, m in zip(all_preds, all_targets, cls_mask) if m)
        cls_acc = (cls_correct / cls_total * 100.0) if cls_total > 0 else 0.0
        bar = "█" * int(cls_acc / 5)
        print(f"  {cls_name:<23} : {cls_correct}/{cls_total:<12} {cls_acc:5.1f}%  {bar}")
    print("-" * 55)

    print("\n[SUCCESS] Session 3 (Training Late Fusion Head) completed.")


if __name__ == "__main__":
    main()
