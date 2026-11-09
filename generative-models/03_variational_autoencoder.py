"""
03_variational_autoencoder.py
=============================
Variational Autoencoder (VAE) Implementation and Latent Manifold Traversal.

Key Concepts (Lecture 13, Slides 71-112):
- Latent variable model with Gaussian prior p(z) = N(0, I).
- Intractable posterior p(z|x) approximated by encoder q_phi(z|x).
- Evidence Lower Bound (ELBO):
    ELBO = E_{z ~ q}[log p(x|z)] - D_KL(q_phi(z|x) || p(z))
- Reparameterization Trick: z = mu + eps * sigma, eps ~ N(0, I).
- Tug-of-war between Reconstruction Loss and Prior Matching Loss.
- Latent Space Disentanglement and 2D Manifold Grid Traversal.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import DEFAULT_SAMPLES_DIR, get_dataloader, get_device, save_image_grid


class VAE(nn.Module):
    """
    Variational Autoencoder with convolutional encoder and decoder.
    """

    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder Network: x -> (mu, log_var)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder Network: z -> x_recon
        self.decoder_fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.Sigmoid(),  # Output Bernoulli means / probabilities in [0, 1]
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: allows gradients to backpropagate through stochastic node.
        z = mu + eps * sigma, where eps ~ N(0, I)
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(-1, 64, 7, 7)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


def elbo_loss(x: torch.Tensor, x_recon: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor):
    """
    Computes the negative ELBO (-ELBO = NLL_recon + KL_div).
    Minimizing -ELBO is equivalent to maximizing the Evidence Lower Bound.
    """
    # 1. Reconstruction Loss: Binary Cross-Entropy sum across pixels
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction="sum")

    # 2. KL Divergence: D_KL(N(mu, sigma^2) || N(0, I))
    # Closed form: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return (recon_loss + kl_div) / x.size(0), recon_loss / x.size(0), kl_div / x.size(0)


@torch.no_grad()
def plot_latent_manifold(model: VAE, device: torch.device, filename: str, n: int = 20):
    """
    Generates a 2D grid manifold by sweeping across latent space (Lecture Slide 112).
    Only valid if latent_dim == 2.
    """
    model.eval()
    import numpy as np
    from scipy.stats import norm

    # Use inverse CDF (percent-point function) for linearly spaced normal coordinates
    grid_x = norm.ppf(np.linspace(0.02, 0.98, n))
    grid_y = norm.ppf(np.linspace(0.02, 0.98, n))

    images = []
    for yi in grid_y:
        for xi in grid_x:
            z = torch.tensor([[xi, yi]], dtype=torch.float32, device=device)
            sample = model.decode(z)
            images.append(sample.squeeze(0))

    images_tensor = torch.stack(images, dim=0)
    save_image_grid(
        images_tensor,
        filename=filename,
        nrow=n,
        title="VAE 2D Latent Manifold Traversal (Disentangling Factors of Variation)",
    )


def train_vae(args):
    device = get_device(args.device)
    print(f"Using compute device: {device}")

    train_loader = get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train=True,
        normalize_range=(0.0, 1.0),
        quick_test=args.quick_test,
    )

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Training Variational Autoencoder (latent_dim={args.latent_dim}) for {args.epochs} epoch(s)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, _ in pbar:
            images = images.to(device)
            optimizer.zero_grad()
            x_recon, mu, log_var = model(images)
            loss, recon, kl = elbo_loss(images, x_recon, mu, log_var)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            pbar.set_postfix({"ELBO": f"{loss.item():.2f}", "Recon": f"{recon.item():.2f}", "KL": f"{kl.item():.2f}"})

        num_batches = len(train_loader)
        print(f"Epoch {epoch} finished | -ELBO: {total_loss / num_batches:.2f} | Recon: {total_recon / num_batches:.2f} | KL: {total_kl / num_batches:.2f}")

    # Generate samples from standard Gaussian prior z ~ N(0, I)
    model.eval()
    with torch.no_grad():
        prior_z = torch.randn(args.num_samples, args.latent_dim, device=device)
        generated_samples = model.decode(prior_z)
        sample_path = os.path.join(args.output_dir, "vae_samples.png")
        save_image_grid(
            generated_samples,
            filename=sample_path,
            nrow=8,
            title="VAE Generated Samples from Prior z ~ N(0, I)",
        )

        # If latent_dim == 2, plot continuous 2D manifold traversal
        if args.latent_dim == 2:
            try:
                import scipy  # noqa: F401
                manifold_path = os.path.join(args.output_dir, "vae_manifold_2d.png")
                plot_latent_manifold(model, device, manifold_path, n=16)
            except ImportError:
                print("Note: Install scipy to view full inverse-CDF manifold traversal.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IF5152: Variational Autoencoders (VAEs)")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=20, help="Set to 2 to generate 2D manifold plot")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--quick-test", action="store_true", help="Quick run on subset for verification")
    args = parser.parse_args()

    train_vae(args)
