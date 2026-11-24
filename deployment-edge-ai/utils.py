"""
Shared utilities for IF5152 Week 14: Deployment Model & Edge AI Laboratory.
Supports NVIDIA CUDA, Apple Silicon (MPS), and CPU.
Provides CIFAR-10 data loaders, sample image generation, model size helpers,
and accurate warmup + synchronized FPS/latency benchmarking.
"""

import os
import time
from typing import Callable, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image, ImageDraw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
DEFAULT_SAMPLES_DIR = os.path.join(SCRIPT_DIR, "samples")
DEFAULT_CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Returns the best available compute device (cuda > mps > cpu) or honors preferred."""
    if preferred:
        pref = preferred.lower()
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref == "cpu":
            return torch.device("cpu")
        print(f"[Device] Requested '{preferred}' unavailable; falling back to auto-select.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    """Synchronizes asynchronous GPU/MPS kernels before stopping a timer."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def get_file_size_mb(path: str) -> float:
    """Returns file size in megabytes."""
    return os.path.getsize(path) / (1024.0 ** 2)


def get_state_dict_size_mb(model: torch.nn.Module) -> float:
    """Measures serialized PyTorch state_dict size in MB via an in-memory buffer."""
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / (1024.0 ** 2)


def count_parameters(model: torch.nn.Module) -> int:
    """Returns total number of parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters())


def get_cifar10_loaders(
    batch_size: int = 64,
    img_size: int = 224,
    train_subset: Optional[int] = None,
    val_subset: Optional[int] = None,
    data_dir: Optional[str] = None,
) -> tuple[DataLoader, DataLoader]:
    """Downloads/loads CIFAR-10 with ImageNet normalization for pretrained torchvision models."""
    root = data_dir or DEFAULT_DATA_DIR
    os.makedirs(root, exist_ok=True)

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
    val_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=val_tf)

    if train_subset is not None and train_subset < len(train_ds):
        train_ds = Subset(train_ds, list(range(train_subset)))
    if val_subset is not None and val_subset < len(val_ds):
        val_ds = Subset(val_ds, list(range(val_subset)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def evaluate_classifier(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Computes Top-1 accuracy (%) on a DataLoader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / max(total, 1)


def benchmark_callable(
    fn: Callable[[], object],
    warmup: int = 15,
    iterations: int = 100,
    sync_fn: Optional[Callable[[], None]] = None,
) -> dict[str, float]:
    """
    Runs warmup iterations followed by timed iterations and returns:
    mean_ms, p95_ms, and fps.
    """
    for _ in range(warmup):
        fn()
    if sync_fn:
        sync_fn()

    latencies_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        if sync_fn:
            sync_fn()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms, dtype=np.float64)
    mean_ms = float(arr.mean())
    p95_ms = float(np.percentile(arr, 95))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return {"mean_ms": mean_ms, "p95_ms": p95_ms, "fps": fps}


def ensure_sample_images(samples_dir: Optional[str] = None, count: int = 3) -> list[str]:
    """
    Ensures sample RGB JPEG images exist in samples/ for ONNX and FastAPI testing.
    Uses CIFAR-10 test images if already downloaded, otherwise generates crisp synthetic RGB images.
    """
    out_dir = samples_dir or DEFAULT_SAMPLES_DIR
    os.makedirs(out_dir, exist_ok=True)
    existing = sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if len(existing) >= count:
        return existing[:count]

    paths: list[str] = []
    cifar_dir = os.path.join(DEFAULT_DATA_DIR, "cifar-10-batches-py")
    if os.path.exists(cifar_dir):
        ds = datasets.CIFAR10(root=DEFAULT_DATA_DIR, train=False, download=False)
        for i in range(count):
            img, label = ds[i]
            img_big = img.resize((224, 224), Image.BILINEAR)
            p = os.path.join(out_dir, f"sample_{i}_{CIFAR10_CLASSES[label]}.jpg")
            img_big.save(p, quality=95)
            paths.append(p)
        return paths


def export_to_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    onnx_path: str,
    opset_version: int = 17,
    input_names: Optional[list[str]] = None,
    output_names: Optional[list[str]] = None,
    dynamic_axes: Optional[dict] = None,
) -> None:
    """Exports a PyTorch model to ONNX using TorchScript tracing (dynamo=False) for broad compatibility."""
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names or ["images"],
        output_names=output_names or ["logits"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )


    colors = [(40, 110, 200), (210, 160, 50), (60, 165, 110)]
    for i in range(count):
        im = Image.new("RGB", (224, 224), color=colors[i % len(colors)])
        draw = ImageDraw.Draw(im)
        draw.rectangle([48, 48, 176, 176], outline=(255, 255, 255), width=4)
        p = os.path.join(out_dir, f"sample_{i}.jpg")
        im.save(p, quality=95)
        paths.append(p)
    return paths


def export_to_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    onnx_path: str,
    opset_version: int = 17,
    input_names: Optional[list[str]] = None,
    output_names: Optional[list[str]] = None,
    dynamic_axes: Optional[dict] = None,
) -> None:
    """Exports a PyTorch model to ONNX using TorchScript tracing (dynamo=False) for broad compatibility."""
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names or ["images"],
        output_names=output_names or ["logits"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

