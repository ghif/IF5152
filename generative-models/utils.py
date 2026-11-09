"""
Shared utilities for IF5152 Generative Models Laboratory.
Supports Apple Silicon (MPS), NVIDIA CUDA, and CPU.
"""

import os
from typing import Optional, Tuple
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Selects the optimal compute device.
    Priority if None: CUDA -> MPS (Apple Silicon) -> CPU.
    """
    if preferred:
        pref = preferred.lower()
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif pref == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif pref == "cpu":
            return torch.device("cpu")
        print(f"[Warning] Requested device '{preferred}' unavailable. Falling back.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
DEFAULT_SAMPLES_DIR = os.path.join(SCRIPT_DIR, "samples")
DEFAULT_CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")


def get_dataloader(
    dataset_name: str = "mnist",
    batch_size: int = 128,
    train: bool = True,
    binary: bool = False,
    normalize_range: Tuple[float, float] = (0.0, 1.0),
    data_dir: Optional[str] = None,
    quick_test: bool = False,
    max_samples: int = 1000,
    lsun_classes: Optional[list] = None,
) -> DataLoader:
    """
    Loads MNIST, Fashion-MNIST, CIFAR-10, or LSUN with customizable normalization.
    For LSUN (Radford et al. 2016): Resizes & CenterCrops to 64x64 with [-1, 1] normalization.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    name = dataset_name.lower()
    is_cifar = name in ["cifar", "cifar10", "cifar-10"]
    is_lsun = name in ["lsun", "lsun-scene", "lsun_scene"]

    if is_lsun:
        # Preprocessing following Radford et al. (2016) Section 4:
        # Resize to 64, CenterCrop 64, HorizontalFlip (for train), and normalize to [-1, 1]
        transform_list = [
            transforms.Resize(64),
            transforms.CenterCrop(64),
        ]
        if train:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.append(transforms.ToTensor())
        if normalize_range == (-1.0, 1.0):
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform_list)

        lsun_dir = os.path.join(data_dir, "lsun")
        # Check if local ImageFolder structure exists (e.g. data/lsun/bedroom/*.jpg)
        has_subdirs = os.path.exists(lsun_dir) and any(
            os.path.isdir(os.path.join(lsun_dir, d)) for d in os.listdir(lsun_dir) if not d.startswith(".")
        )

        if has_subdirs:
            ds = datasets.ImageFolder(root=lsun_dir, transform=transform)
        else:
            # Check torchvision LMDB LSUN
            try:
                classes_to_load = lsun_classes or (["bedroom_train"] if train else ["bedroom_val"])
                ds = datasets.LSUN(root=lsun_dir, classes=classes_to_load, transform=transform)
            except Exception:
                # Mock fallback dataset for rapid testing / development without 100GB download
                mock_dir = os.path.join(data_dir, "lsun_mock")
                mock_cats = ["bedroom", "church_outdoor", "kitchen"]
                for cat in mock_cats:
                    cat_dir = os.path.join(mock_dir, cat)
                    os.makedirs(cat_dir, exist_ok=True)
                    if len(os.listdir(cat_dir)) < 10:
                        from PIL import Image
                        import numpy as np
                        for idx in range(10):
                            arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
                            Image.fromarray(arr).save(os.path.join(cat_dir, f"sample_{idx}.jpg"))
                ds = datasets.ImageFolder(root=mock_dir, transform=transform)

    else:
        transform_list = [transforms.ToTensor()]
        if binary:
            # Binarize inputs (useful for binary PixelCNN)
            transform_list.append(transforms.Lambda(lambda x: (x > 0.5).float()))
        elif normalize_range == (-1.0, 1.0):
            # Normalize to [-1, 1] (standard for GANs / Diffusion)
            if is_cifar:
                transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            else:
                transform_list.append(transforms.Normalize((0.5,), (0.5,)))

        transform = transforms.Compose(transform_list)

        if name == "fashion":
            ds = datasets.FashionMNIST(root=data_dir, train=train, download=True, transform=transform)
        elif is_cifar:
            ds = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
        else:
            ds = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)

    if quick_test:
        subset_indices = list(range(min(max_samples, len(ds))))
        ds = Subset(ds, subset_indices)

    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train)


def save_image_grid(
    images: torch.Tensor,
    filename: str,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Saves a batch of image tensors as an image grid file using torchvision.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    grid = torchvision.utils.make_grid(
        images.detach().cpu(),
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
        padding=2,
    )
    n_images = images.size(0)
    n_rows = (n_images + nrow - 1) // nrow
    fig_w = max(6.0, nrow * 1.2)
    fig_h = max(2.5, n_rows * 1.2 + (0.8 if title else 0.0))
    plt.figure(figsize=(fig_w, fig_h))
    if title:
        plt.title(title, fontsize=14, pad=10)
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray" if grid.shape[0] == 1 else None)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Saved] Image grid saved to: {filename}")
