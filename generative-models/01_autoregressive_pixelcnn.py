"""
01_autoregressive_pixelcnn.py
=============================
Autoregressive Generative Modeling on Images with PixelCNN.

Key Concepts:
- Exact Maximum Likelihood Estimation via the Chain Rule of Probability.
- Causality via Masked Convolutions (Mask Type A vs Mask Type B).
- Scanline raster-order autoregressive sampling: O(H * W) forward passes.
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import DEFAULT_SAMPLES_DIR, get_dataloader, get_device, save_image_grid


class MaskedConv2d(nn.Conv2d):
    """
    2D Convolution with causal masking for autoregressive modeling.
    - Type 'A': Center pixel is zeroed out (used in the first layer to prevent self-conditioning).
    - Type 'B': Center pixel is retained (used in subsequent hidden layers).
    """

    def __init__(self, mask_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ["A", "B"], "mask_type must be 'A' or 'B'"
        self.mask_type = mask_type

        # Register mask buffer (not a learnable parameter)
        self.register_buffer("mask", torch.ones_like(self.weight))
        _, _, kh, kw = self.weight.shape
        center_h = kh // 2
        center_w = kw // 2

        # Zero out future rows
        self.mask[:, :, center_h + 1 :, :] = 0
        # Zero out future pixels on the center row
        self.mask[:, :, center_h, center_w + 1 :] = 0

        # Type A mask also zeros out the center pixel itself
        if mask_type == "A":
            self.mask[:, :, center_h, center_w] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class GatedResidualBlock(nn.Module):
    """
    Gated convolutional layer with residual connection and Type B masking.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = MaskedConv2d("B", channels, 2 * channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        val, gate = torch.chunk(h, 2, dim=1)
        out = torch.tanh(val) * torch.sigmoid(gate)
        return x + out


class PixelCNN(nn.Module):
    """
    PixelCNN architecture for binary / grayscale images.
    Predicts p(x_t | x_{<t}) for each pixel.
    """

    def __init__(self, in_channels: int = 1, hidden_channels: int = 64, num_layers: int = 6):
        super().__init__()
        # First layer uses Mask A (cannot condition on current pixel)
        self.entry_conv = nn.Sequential(
            MaskedConv2d("A", in_channels, hidden_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
        )

        # Subsequent layers use Gated Residual Blocks with Mask B
        self.blocks = nn.ModuleList([GatedResidualBlock(hidden_channels) for _ in range(num_layers)])

        # Output head predicting Bernoulli logits for each pixel
        self.head = nn.Sequential(
            MaskedConv2d("B", hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            MaskedConv2d("B", hidden_channels, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.entry_conv(x)
        for block in self.blocks:
            h = block(h)
        logits = self.head(h)
        return logits


@torch.no_grad()
def sample_pixelcnn(
    model: nn.Module,
    device: torch.device,
    num_samples: int = 64,
    img_shape: tuple = (1, 28, 28),
) -> torch.Tensor:
    """
    Generates new samples autoregressively, pixel by pixel, in raster scanline order.
    Demonstrates the sequential O(H*W) sampling latency.
    """
    model.eval()
    samples = torch.zeros((num_samples, *img_shape), device=device)
    _, h, w = img_shape

    print(f"Generating {num_samples} samples pixel-by-pixel ({h}x{w} = {h*w} sequential steps)...")
    start_time = time.time()

    for i in range(h):
        for j in range(w):
            logits = model(samples)
            probs = torch.sigmoid(logits[:, :, i, j])
            # Sample Bernoulli for binary MNIST
            sampled_pixel = torch.bernoulli(probs)
            samples[:, :, i, j] = sampled_pixel

    elapsed = time.time() - start_time
    print(f"Autoregressive generation completed in {elapsed:.2f} seconds.")
    return samples


def train_pixelcnn(args):
    device = get_device(args.device)
    print(f"Using compute device: {device}")

    train_loader = get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train=True,
        binary=True,
        quick_test=args.quick_test,
    )

    model = PixelCNN(in_channels=1, hidden_channels=args.hidden_channels, num_layers=args.num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Starting PixelCNN training for {args.epochs} epoch(s)...")
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"NLL Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} finished. Average NLL Loss: {avg_loss:.4f} (Bits/dim: {avg_loss / 0.6931:.2f})")

    # Sample and visualize
    samples = sample_pixelcnn(model, device, num_samples=args.num_samples)
    out_path = os.path.join(args.output_dir, "pixelcnn_samples.png")
    save_image_grid(
        samples,
        filename=out_path,
        nrow=8,
        title="Autoregressive PixelCNN Generated Samples",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IF5152: Autoregressive Image Modeling with PixelCNN")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--quick-test", action="store_true", help="Quick run on subset for verification")
    args = parser.parse_args()

    train_pixelcnn(args)
