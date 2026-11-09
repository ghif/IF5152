"""
04a_generative_adversarial_network.py
====================================
Unified Deep Convolutional Generative Adversarial Network (DCGAN).
Supports MNIST (1x28x28), Fashion-MNIST (1x28x28), and CIFAR-10 (3x32x32).

Key Concepts & Architecture Guidelines (Radford et al., ICLR 2016; Lecture 13):
1. Implicit Density: Does NOT compute p(x), directly samples G(z) ~ p_g in O(1).
2. Two-Player Minimax Game:
     min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]
3. Non-saturating Generator loss: max_G log D(G(z)) to eliminate vanishing gradients early.
4. All-convolutional design: strided convolutions in Discriminator, transposed convolutions in Generator.
5. Spatial Batch Normalization in both networks (omitted on Discriminator input & Generator output).
6. LeakyReLU (slope 0.2) in Discriminator, ReLU in Generator, Tanh output (normalized to [-1, 1]).
7. Weight initialization: Normal N(0, 0.02^2).
8. Fixed noise probe evaluated and saved to gan_samples.png after each end of epoch.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import DEFAULT_SAMPLES_DIR, get_dataloader, get_device, save_image_grid

DEFAULT_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def weights_init(m: nn.Module) -> None:
    """
    Custom DCGAN weight initialization:
    - Convolutional weights initialized from N(0.0, 0.02^2).
    - BatchNorm weights initialized from N(1.0, 0.02^2), bias set to 0.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ==============================================================================
# 28x28 Architecture (MNIST / Fashion-MNIST, 1 Channel)
# ==============================================================================

class Generator28(nn.Module):
    """
    DCGAN Generator for 1x28x28 images (MNIST / Fashion-MNIST).
    Projects noise vector z through a dense projection to 7x7 followed by transposed convs.
    """

    def __init__(self, latent_dim: int = 100, ngf: int = 64, channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.ngf = ngf

        self.project = nn.Sequential(
            nn.Linear(latent_dim, (ngf * 2) * 7 * 7),
            nn.BatchNorm1d((ngf * 2) * 7 * 7),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(ngf, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),  # Normalized to [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 4:
            z = z.flatten(1)
        h = self.project(z).view(-1, self.ngf * 2, 7, 7)
        return self.conv(h)


class Discriminator28(nn.Module):
    """
    DCGAN Discriminator for 1x28x28 images (MNIST / Fashion-MNIST).
    Maps 1x28x28 image to probability of being real.
    """

    def __init__(self, ndf: int = 64, channels: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            # 28x28 -> 14x14 (No BatchNorm on discriminator input layer)
            nn.Conv2d(channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 14x14 -> 7x7
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear((ndf * 2) * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ==============================================================================
# 32x32 Architecture (CIFAR-10, 3 Channels RGB)
# ==============================================================================

class Generator32(nn.Module):
    """
    DCGAN Generator for 3x32x32 RGB images (CIFAR-10, Radford et al. 2016).
    Maps noise vector z in R^{latent_dim x 1 x 1} to synthetic image 3x32x32 in [-1, 1].
    """

    def __init__(self, latent_dim: int = 100, ngf: int = 64, channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            # z (latent_dim x 1 x 1) -> (ngf * 4) x 4 x 4
            nn.ConvTranspose2d(latent_dim, ngf * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            # (ngf * 4) x 4 x 4 -> (ngf * 2) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            # (ngf * 2) x 8 x 8 -> ngf x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # ngf x 16 x 16 -> channels x 32 x 32
            nn.ConvTranspose2d(ngf, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),  # Normalized to [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        return self.net(z)


class Discriminator32(nn.Module):
    """
    DCGAN Discriminator for 3x32x32 RGB images (CIFAR-10, Radford et al. 2016).
    Maps image 3x32x32 to a scalar probability in [0, 1] of being real.
    """

    def __init__(self, ndf: int = 64, channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            # channels x 32 x 32 -> ndf x 16 x 16 (No BatchNorm on input layer)
            nn.Conv2d(channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf x 16 x 16 -> (ndf * 2) x 8 x 8
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 2) x 8 x 8 -> (ndf * 4) x 4 x 4
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 4) x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_dcgan_models(dataset: str, latent_dim: int = 100, ngf: int = 64, ndf: int = 64):
    """
    Factory function returning the appropriate Generator, Discriminator,
    and latent probe shape for the chosen dataset.
    """
    ds = dataset.lower()
    if ds in ["cifar", "cifar10", "cifar-10"]:
        netG = Generator32(latent_dim=latent_dim, ngf=ngf, channels=3)
        netD = Discriminator32(ndf=ndf, channels=3)
        sample_shape = (latent_dim, 1, 1)
    else:
        netG = Generator28(latent_dim=latent_dim, ngf=ngf, channels=1)
        netD = Discriminator28(ndf=ndf, channels=1)
        sample_shape = (latent_dim,)

    netG.apply(weights_init)
    netD.apply(weights_init)
    return netG, netD, sample_shape


# ==============================================================================
# Training Procedure
# ==============================================================================

def train_gan(args):
    device = get_device(args.device)
    print(f"Using compute device: {device}")

    # GANs require [-1, 1] normalization to pair with Tanh activation
    train_loader = get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train=True,
        normalize_range=(-1.0, 1.0),
        quick_test=args.quick_test,
    )

    netG, netD, sample_shape = get_dcgan_models(
        dataset=args.dataset,
        latent_dim=args.latent_dim,
        ngf=args.ngf,
        ndf=args.ndf,
    )
    netG = netG.to(device)
    netD = netD.to(device)

    # Heuristic: Adam with beta1=0.5 stabilizes GAN training
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    criterion = nn.BCELoss()

    # Fixed noise probe to monitor generated samples across epochs
    fixed_noise = torch.randn(args.num_samples, *sample_shape, device=device)

    print(f"Training Generative Adversarial Network on {args.dataset.upper()} for {args.epochs} epoch(s)...")
    for epoch in range(1, args.epochs + 1):
        netD.train()
        netG.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for real_images, _ in pbar:
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # One-sided label smoothing for discriminator real targets (Salimans et al. 2016)
            real_labels = torch.full((batch_size, 1), args.label_smoothing, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ==============================
            # 1. Update Discriminator (netD)
            # Maximize log(D(x)) + log(1 - D(G(z)))
            # ==============================
            optimizerD.zero_grad()

            # Loss on real images
            output_real = netD(real_images)
            loss_d_real = criterion(output_real, real_labels)

            # Loss on fake images
            noise = torch.randn(batch_size, *sample_shape, device=device)
            fake_images = netG(noise)
            output_fake = netD(fake_images.detach())
            loss_d_fake = criterion(output_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizerD.step()

            # ==============================
            # 2. Update Generator (netG)
            # Maximize log(D(G(z))) (Non-saturating heuristic, target=1.0)
            # ==============================
            optimizerG.zero_grad()
            output_fake_for_g = netD(fake_images)
            loss_g = criterion(output_fake_for_g, torch.ones(batch_size, 1, device=device))
            loss_g.backward()
            optimizerG.step()

            pbar.set_postfix({
                "Loss D": f"{loss_d.item():.4f}",
                "Loss G": f"{loss_g.item():.4f}",
                "D(x)": f"{output_real.mean().item():.2f}",
                "D(G(z))": f"{output_fake.mean().item():.2f}",
            })

        # ==============================
        # Generate samples after each epoch
        # ==============================
        netG.eval()
        with torch.no_grad():
            fake_samples = netG(fixed_noise)
            out_path = os.path.join(args.output_dir, "gan_samples.png")
            save_image_grid(
                fake_samples,
                filename=out_path,
                nrow=8,
                value_range=(-1.0, 1.0),
                title=f"DCGAN ({args.dataset.upper()}) Samples (Epoch {epoch}/{args.epochs})",
            )
            dataset_out = os.path.join(args.output_dir, f"gan_samples_{args.dataset.lower()}.png")
            save_image_grid(
                fake_samples,
                filename=dataset_out,
                nrow=8,
                value_range=(-1.0, 1.0),
                title=f"DCGAN ({args.dataset.upper()}) Samples (Epoch {epoch}/{args.epochs})",
            )

            # Save milestone progression snapshots
            if args.save_milestones and (epoch in [1, 5, 10, 15, 20, 25, 30] or epoch == args.epochs):
                milestone_dir = os.path.join(args.output_dir, f"{args.dataset.lower()}_milestones")
                os.makedirs(milestone_dir, exist_ok=True)
                milestone_path = os.path.join(milestone_dir, f"epoch_{epoch:02d}.png")
                save_image_grid(
                    fake_samples,
                    filename=milestone_path,
                    nrow=8,
                    value_range=(-1.0, 1.0),
                    title=f"DCGAN ({args.dataset.upper()}) Milestone Epoch {epoch}/{args.epochs}",
                )

    # Save model checkpoints
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    gen_ckpt = os.path.join(args.checkpoint_dir, f"dcgan_{args.dataset.lower()}_generator.pt")
    disc_ckpt = os.path.join(args.checkpoint_dir, f"dcgan_{args.dataset.lower()}_discriminator.pt")
    torch.save(netG.state_dict(), gen_ckpt)
    torch.save(netD.state_dict(), disc_ckpt)
    print(f"[Saved] Generator checkpoint saved to: {gen_ckpt}")
    print(f"[Saved] Discriminator checkpoint saved to: {disc_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IF5152: Generative Adversarial Networks (DCGAN)")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion", "cifar10"],
                        help="Dataset to train on (mnist, fashion, cifar10)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--latent-dim", type=int, default=100, help="Latent noise vector dimension")
    parser.add_argument("--ngf", type=int, default=64, help="Generator feature maps multiplier")
    parser.add_argument("--ndf", type=int, default=64, help="Discriminator feature maps multiplier")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for Adam")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam momentum beta1")
    parser.add_argument("--label-smoothing", type=float, default=0.9, help="One-sided label smoothing for real images")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda/mps/cpu)")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples to visualize in grid")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_SAMPLES_DIR, help="Directory to save output samples")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="Directory to save checkpoints")
    parser.add_argument("--save-milestones", action="store_true", default=True, help="Save milestone snapshots")
    parser.add_argument("--quick-test", action="store_true", help="Quick run on subset for rapid testing")
    args = parser.parse_args()

    train_gan(args)
