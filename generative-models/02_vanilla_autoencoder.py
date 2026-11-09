"""
02_vanilla_autoencoder.py
=========================
Classical (Deterministic) Autoencoders and the Limits of Latent Generation.

Key Concepts (Lecture 13, Slides 63-70):
- Unsupervised representation learning via input reconstruction: ||x_hat - x||^2.
- Deterministic encoder z = f_enc(x) and decoder x_hat = f_dec(z).
- Why Vanilla Autoencoders are NOT Generative Models:
  The latent space is unconstrained and full of 'holes' (discontinuous).
  Sampling z ~ N(0, I) leads to unnatural, garbled artifacts.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import DEFAULT_SAMPLES_DIR, get_dataloader, get_device, save_image_grid


class VanillaAutoencoder(nn.Module):
    """
    Convolutional Autoencoder with low-dimensional bottleneck latent space.
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 1x28x28 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim),
        )

        # Decoder: latent_dim -> 1x28x28
        self.decoder_fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.Sigmoid(),  # Pixel values in [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(-1, 64, 7, 7)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


def train_autoencoder(args):
    device = get_device(args.device)
    print(f"Using compute device: {device}")

    train_loader = get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train=True,
        normalize_range=(0.0, 1.0),
        quick_test=args.quick_test,
    )
    test_loader = get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train=False,
        normalize_range=(0.0, 1.0),
        quick_test=args.quick_test,
    )

    model = VanillaAutoencoder(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Training Classical Autoencoder for {args.epochs} epoch(s)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, _ in pbar:
            images = images.to(device)
            optimizer.zero_grad()
            reconstructions = model(images)
            loss = criterion(reconstructions, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"Reconstruction MSE": f"{loss.item():.5f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} finished. Average MSE: {avg_loss:.5f}")

    # Demonstration 1: Sharp Reconstructions
    model.eval()
    with torch.no_grad():
        test_images, _ = next(iter(test_loader))
        test_images = test_images[:16].to(device)
        recon_images = model(test_images)

        # Interleave originals and reconstructions
        comparison = torch.cat([test_images, recon_images], dim=0)
        recon_path = os.path.join(args.output_dir, "ae_reconstructions.png")
        save_image_grid(
            comparison,
            filename=recon_path,
            nrow=16,
            title="Vanilla AE: Top row = Original, Bottom row = Reconstructed",
        )

        # Demonstration 2: Failure of Naive Latent Sampling (Lecture Slide 69-70)
        print("\nDemonstrating why Vanilla AE fails as a generative model...")
        random_z = torch.randn(32, args.latent_dim, device=device)
        failed_samples = model.decode(random_z)
        failure_path = os.path.join(args.output_dir, "ae_random_sampling_failure.png")
        save_image_grid(
            failed_samples,
            filename=failure_path,
            nrow=8,
            title="Vanilla AE Failure: Random z ~ N(0, I) yields unconstrained/garbled artifacts",
        )
        print("Done! Notice how arbitrary sampling fails without a regularized prior distribution.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IF5152: Classical Autoencoder and Latent Generation Limitations")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--quick-test", action="store_true", help="Quick run on subset for verification")
    args = parser.parse_args()

    train_autoencoder(args)
