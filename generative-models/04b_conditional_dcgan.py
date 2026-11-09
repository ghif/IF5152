"""
04b_conditional_dcgan.py
========================
Class-Conditional Deep Convolutional Generative Adversarial Network (cGAN).
Supports:
  1. CIFAR-10 (3x32x32 RGB images across 10 classes)
  2. LSUN (3x64x64 RGB images across scene categories, exactly following Radford et al., ICLR 2016)

Key Concepts & Architecture Guidelines (Radford et al. 2016; Mirza & Osindero 2014):
1. Conditional Minimax Game:
     min_G max_D V(D, G) = E_{(x, y)}[log D(x, y)] + E_{z, y}[log(1 - D(G(z, y), y))]
2. Generator Conditioning:
     Class label y is mapped to an embedding vector and concatenated with
     the latent noise vector z before transposed conv upsampling.
3. Discriminator Conditioning:
     Class label y is mapped to an img_size x img_size spatial feature channel
     and concatenated as a 4th channel to the RGB image (4 x H x W).
4. Radford et al. (2016) 64x64 LSUN Architecture:
     Four fractionally-strided transposed convolutions (4 -> 8 -> 16 -> 32 -> 64)
     with feature maps 512 -> 256 -> 128 -> 64 -> 3 and Tanh activation (Figure 1).
5. Automatic Checkpoint Persistence & Class Grid Probes.
"""

import argparse
import os
import sys
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import DEFAULT_CHECKPOINT_DIR, DEFAULT_SAMPLES_DIR, get_dataloader, get_device, save_image_grid

# Dataset default class definitions
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

DEFAULT_LSUN_CATEGORIES = [
    "bedroom", "church_outdoor", "kitchen", "living_room",
    "bridge", "classroom", "conference_room", "dining_room", "restaurant", "tower",
]


def weights_init(m: nn.Module) -> None:
    """
    DCGAN Weight Initialization (Radford et al. 2016):
    - Conv & ConvTranspose weights initialized from N(0.0, 0.02^2).
    - BatchNorm weights initialized from N(1.0, 0.02^2), bias set to 0.0.
    - Class Embedding weights initialized from N(0.0, 0.02^2) to prevent spatial signal corruption.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Embedding") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ==============================================================================
# Model Architectures: 32x32 (CIFAR-10)
# ==============================================================================

class ConditionalGenerator32(nn.Module):
    """
    Class-Conditional DCGAN Generator for 3x32x32 RGB images (CIFAR-10).
    Uses 3 transposed convolution upsampling stages: 4 -> 8 -> 16 -> 32.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        embed_dim: int = 100,
        num_classes: int = 10,
        ngf: int = 64,
        channels: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.ngf = ngf
        self.channels = channels

        self.label_emb = nn.Embedding(num_classes, embed_dim)
        in_channels = latent_dim + embed_dim

        self.net = nn.Sequential(
            # in_channels x 1 x 1 -> (ngf * 4) x 4 x 4
            nn.ConvTranspose2d(in_channels, ngf * 4, kernel_size=4, stride=1, padding=0, bias=False),
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
            nn.Tanh(),  # Normalized output in [-1, 1]
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if z.dim() == 4:
            z = z.flatten(1)
        y_emb = self.label_emb(y)
        z_cond = torch.cat([z, y_emb], dim=1).unsqueeze(-1).unsqueeze(-1)
        return self.net(z_cond)


class ConditionalDiscriminator32(nn.Module):
    """
    Class-Conditional DCGAN Discriminator for 3x32x32 RGB images (CIFAR-10).
    Uses 3 strided convolution stages: 32 -> 16 -> 8 -> 4 -> 1.
    Spatially broadcasts class embedding (1 channel) across the image to avoid grid noise.
    """

    def __init__(
        self,
        num_classes: int = 10,
        ndf: int = 64,
        channels: int = 3,
        img_size: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ndf = ndf
        self.channels = channels
        self.img_size = img_size

        self.label_emb = nn.Embedding(num_classes, 1)
        in_channels = channels + 1

        self.net = nn.Sequential(
            # 4 x 32 x 32 -> ndf x 16 x 16
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        y_map = self.label_emb(y).view(b, 1, 1, 1).expand(b, 1, h, w)
        x_cond = torch.cat([x, y_map], dim=1)
        return self.net(x_cond)


# ==============================================================================
# Model Architectures: 64x64 (LSUN Scene Modeling, Radford et al. 2016 Fig. 1)
# ==============================================================================

class ConditionalGenerator64(nn.Module):
    """
    Class-Conditional DCGAN Generator for 3x64x64 RGB images (LSUN Scene Modeling).
    Supports two upsampling mechanisms:
    1. 'conv_transpose': Figure 1 of Radford et al. (ICLR 2016) fractionally-strided convs (512->256->128->64->3).
    2. 'nearest_conv': Anti-checkerboard upsampling (Odena et al. 2016) via Nearest Upsample + Conv2d.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        embed_dim: int = 100,
        num_classes: int = 10,
        ngf: int = 64,
        channels: int = 3,
        upsample_mode: str = "nearest_conv",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.ngf = ngf
        self.channels = channels
        self.upsample_mode = upsample_mode

        self.label_emb = nn.Embedding(num_classes, embed_dim)
        in_channels = latent_dim + embed_dim

        if upsample_mode == "conv_transpose":
            self.net = nn.Sequential(
                # in_channels x 1 x 1 -> (ngf * 8) x 4 x 4 (Radford et al.: 512 x 4 x 4)
                nn.ConvTranspose2d(in_channels, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(inplace=True),
                # (ngf * 8) x 4 x 4 -> (ngf * 4) x 8 x 8 (Radford et al.: 256 x 8 x 8)
                nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(inplace=True),
                # (ngf * 4) x 8 x 8 -> (ngf * 2) x 16 x 16 (Radford et al.: 128 x 16 x 16)
                nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(inplace=True),
                # (ngf * 2) x 16 x 16 -> ngf x 32 x 32 (Radford et al.: 64 x 32 x 32)
                nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(inplace=True),
                # ngf x 32 x 32 -> channels x 64 x 64 (Radford et al.: 3 x 64 x 64)
                nn.ConvTranspose2d(ngf, channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh(),
            )
        else:
            # Anti-checkerboard pipeline (Odena et al. 2016): Nearest Upsampling + Conv2d
            self.net = nn.Sequential(
                nn.ConvTranspose2d(in_channels, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(ngf, channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh(),
            )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if z.dim() == 4:
            z = z.flatten(1)
        y_emb = self.label_emb(y)
        z_cond = torch.cat([z, y_emb], dim=1).unsqueeze(-1).unsqueeze(-1)
        return self.net(z_cond)


class ConditionalDiscriminator64(nn.Module):
    """
    Class-Conditional DCGAN Discriminator for 3x64x64 RGB images (LSUN Scene Modeling).
    Four strided convolution stages: 64 -> 32 -> 16 -> 8 -> 4 -> 1.
    Spatially broadcasts class embedding (1 channel) across the image to avoid grid noise.
    """

    def __init__(
        self,
        num_classes: int = 10,
        ndf: int = 64,
        channels: int = 3,
        img_size: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ndf = ndf
        self.channels = channels
        self.img_size = img_size

        self.label_emb = nn.Embedding(num_classes, 1)
        in_channels = channels + 1

        self.net = nn.Sequential(
            # 4 x 64 x 64 -> ndf x 32 x 32 (No BatchNorm on input layer)
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf x 32 x 32 -> (ndf * 2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 2) x 16 x 16 -> (ndf * 4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 4) x 8 x 8 -> (ndf * 8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 8) x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.Sigmoid(),  # Probability in [0, 1]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        y_map = self.label_emb(y).view(b, 1, 1, 1).expand(b, 1, h, w)
        x_cond = torch.cat([x, y_map], dim=1)
        return self.net(x_cond)


def get_conditional_dcgan_models(
    dataset: str,
    img_size: int,
    num_classes: int,
    latent_dim: int = 100,
    embed_dim: int = 100,
    ngf: int = 64,
    ndf: int = 64,
    channels: int = 3,
    upsample_mode: str = "nearest_conv",
) -> Tuple[nn.Module, nn.Module]:
    """
    Factory creating the appropriate conditional generator and discriminator.
    """
    if img_size == 64 or dataset.lower() == "lsun":
        netG = ConditionalGenerator64(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
            ngf=ngf,
            channels=channels,
            upsample_mode=upsample_mode,
        )
        netD = ConditionalDiscriminator64(
            num_classes=num_classes,
            ndf=ndf,
            channels=channels,
            img_size=64,
        )
    else:
        netG = ConditionalGenerator32(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
            ngf=ngf,
            channels=channels,
        )
        netD = ConditionalDiscriminator32(
            num_classes=num_classes,
            ndf=ndf,
            channels=channels,
            img_size=32,
        )

    netG.apply(weights_init)
    netD.apply(weights_init)
    return netG, netD


def create_fixed_conditional_probe(
    device: torch.device,
    num_classes: int,
    latent_dim: int,
    samples_per_class: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a reproducible (num_classes x samples_per_class) evaluation grid probe.
    """
    torch.manual_seed(42)
    base_noise = torch.randn(samples_per_class, latent_dim, device=device)
    fixed_noise = base_noise.repeat(num_classes, 1)
    fixed_labels = torch.arange(num_classes, device=device).repeat_interleave(samples_per_class)
    return fixed_noise, fixed_labels


# ==============================================================================
# Training Pipeline
# ==============================================================================

def train_cgan(args):
    device = get_device(args.device)
    dataset = args.dataset.lower()

    # Determine resolution
    if dataset == "lsun":
        img_size = args.img_size if args.img_size is not None else 64
        lsun_cats = [c.strip() for c in args.lsun_categories.split(",") if c.strip()]
        class_names = lsun_cats if lsun_cats else DEFAULT_LSUN_CATEGORIES
    else:
        img_size = args.img_size if args.img_size is not None else 32
        class_names = CIFAR10_CLASSES

    # 1. Instantiate Data Loader
    train_loader = get_dataloader(
        dataset_name=dataset,
        batch_size=args.batch_size,
        train=True,
        normalize_range=(-1.0, 1.0),
        data_dir=args.data_dir,
        quick_test=args.quick_test,
        max_samples=500,
        lsun_classes=[f"{c}_train" for c in class_names] if dataset == "lsun" else None,
    )

    # Detect dataset classes if available (e.g. from ImageFolder)
    base_ds = train_loader.dataset
    if hasattr(base_ds, "dataset"):
        base_ds = base_ds.dataset  # unwrap Subset if quick_test
    if hasattr(base_ds, "classes") and base_ds.classes:
        class_names = base_ds.classes

    num_classes = len(class_names)

    print("=" * 65)
    print(f"IF5152: Class-Conditional DCGAN Training ({dataset.upper()})")
    print(f"Device        : {device}")
    print(f"Dataset       : {dataset} (Resolution: {img_size}x{img_size}, Channels: 3)")
    print(f"Num Classes   : {num_classes} ({', '.join(class_names)})")
    print(f"Latent Dim    : {args.latent_dim} | Embedding Dim : {args.embed_dim}")
    print(f"Feature Maps  : ngf={args.ngf}, ndf={args.ndf}")
    print(f"Upsample Mode : {args.upsample_mode}")
    print(f"Batch Size    : {args.batch_size} | Epochs : {args.epochs}")
    print(f"Learning Rate : {args.lr} | Beta1 : {args.beta1}")
    print(f"Batches/Epoch : {len(train_loader)}")
    print("=" * 65)

    # 2. Instantiate Models
    netG, netD = get_conditional_dcgan_models(
        dataset=dataset,
        img_size=img_size,
        num_classes=num_classes,
        latent_dim=args.latent_dim,
        embed_dim=args.embed_dim,
        ngf=args.ngf,
        ndf=args.ndf,
        channels=3,
        upsample_mode=args.upsample_mode,
    )
    netG.to(device)
    netD.to(device)

    # Checkpoint paths
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    full_ckpt_path = os.path.join(args.checkpoint_dir, f"cgan_{dataset}_checkpoint.pt")
    gen_ckpt_path = os.path.join(args.checkpoint_dir, f"cgan_{dataset}_generator.pt")
    disc_ckpt_path = os.path.join(args.checkpoint_dir, f"cgan_{dataset}_discriminator.pt")

    # Optimizers & Loss
    start_epoch = 1
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    criterion = nn.BCELoss()

    # Resume if requested
    if args.resume and os.path.exists(full_ckpt_path):
        print(f"[Resume] Loading checkpoint from {full_ckpt_path}...")
        ckpt = torch.load(full_ckpt_path, map_location=device)
        netG.load_state_dict(ckpt["generator_state_dict"])
        netD.load_state_dict(ckpt["discriminator_state_dict"])
        optimizerG.load_state_dict(ckpt["optimizerG_state_dict"])
        optimizerD.load_state_dict(ckpt["optimizerD_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] Resuming training at epoch {start_epoch}.")

    # Fixed evaluation probe (num_classes x 10 samples)
    fixed_noise, fixed_labels = create_fixed_conditional_probe(
        device=device,
        num_classes=num_classes,
        latent_dim=args.latent_dim,
        samples_per_class=10,
    )

    # Sample-only mode
    if args.sample_only:
        if not os.path.exists(gen_ckpt_path) and not os.path.exists(full_ckpt_path):
            print(f"[Error] No checkpoint found in {args.checkpoint_dir} for {dataset}. Train the model first.")
            sys.exit(1)
        if os.path.exists(gen_ckpt_path):
            netG.load_state_dict(torch.load(gen_ckpt_path, map_location=device))
        else:
            ckpt = torch.load(full_ckpt_path, map_location=device)
            netG.load_state_dict(ckpt["generator_state_dict"])

        netG.eval()
        with torch.no_grad():
            samples = netG(fixed_noise, fixed_labels)
            out_path = os.path.join(args.output_dir, f"cgan_{dataset}_samples.png")
            save_image_grid(
                samples,
                filename=out_path,
                nrow=10,
                value_range=(-1.0, 1.0),
                title=f"Class-Conditional DCGAN ({dataset.upper()} {img_size}x{img_size}) Samples",
            )
            print(f"[Sample Only] Generated class grid saved to: {out_path}")
        return

    # 2. Data Loader
    train_loader = get_dataloader(
        dataset_name=dataset,
        batch_size=args.batch_size,
        train=True,
        normalize_range=(-1.0, 1.0),
        data_dir=args.data_dir,
        quick_test=args.quick_test,
        max_samples=500,
        lsun_classes=[f"{c}_train" for c in class_names] if dataset == "lsun" else None,
    )

    print(f"Training batches per epoch: {len(train_loader)}")

    # 3. Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        netG.train()
        netD.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for real_imgs, real_labels in pbar:
            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            # Ensure label indices are within valid range [0, num_classes - 1]
            real_labels = (real_labels % num_classes).to(device)

            label_real = torch.full((b_size, 1), args.label_smoothing, device=device)
            label_fake = torch.zeros((b_size, 1), device=device)

            # ------------------------------------------------------------------
            # (1) Update Discriminator: max log D(x, y) + log(1 - D(G(z, y), y))
            # ------------------------------------------------------------------
            optimizerD.zero_grad()
            out_real = netD(real_imgs, real_labels)
            loss_d_real = criterion(out_real, label_real)

            noise = torch.randn(b_size, args.latent_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (b_size,), device=device)
            fake_imgs = netG(noise, gen_labels)

            out_fake = netD(fake_imgs.detach(), gen_labels)
            loss_d_fake = criterion(out_fake, label_fake)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizerD.step()

            # ------------------------------------------------------------------
            # (2) Update Generator: max log D(G(z, y), y)
            # ------------------------------------------------------------------
            optimizerG.zero_grad()
            out_fake_g = netD(fake_imgs, gen_labels)
            loss_g = criterion(out_fake_g, torch.ones((b_size, 1), device=device))
            loss_g.backward()
            optimizerG.step()

            pbar.set_postfix({
                "Loss_D": f"{loss_d.item():.4f}",
                "Loss_G": f"{loss_g.item():.4f}",
                "D(x)": f"{out_real.mean().item():.2f}",
                "D(G(z))": f"{out_fake.mean().item():.2f}",
            })

        # ==================================================================
        # Sample Visualizations (Row i = Class i)
        # ==================================================================
        netG.eval()
        with torch.no_grad():
            sample_grid = netG(fixed_noise, fixed_labels)
            out_path = os.path.join(args.output_dir, f"cgan_{dataset}_samples.png")
            save_image_grid(
                sample_grid,
                filename=out_path,
                nrow=10,
                value_range=(-1.0, 1.0),
                title=f"Conditional DCGAN ({dataset.upper()} {img_size}x{img_size}) Epoch {epoch}/{args.epochs}",
            )

            # Milestone checkpoints
            if args.save_milestones and (epoch in [1, 5, 10, 15, 20, 25, 30] or epoch == args.epochs):
                milestone_dir = os.path.join(args.output_dir, f"cgan_{dataset}_milestones")
                os.makedirs(milestone_dir, exist_ok=True)
                milestone_path = os.path.join(milestone_dir, f"epoch_{epoch:02d}.png")
                save_image_grid(
                    sample_grid,
                    filename=milestone_path,
                    nrow=10,
                    value_range=(-1.0, 1.0),
                    title=f"cGAN {dataset.upper()} Milestone (Epoch {epoch}/{args.epochs})",
                )

        # ==================================================================
        # Checkpointing
        # ==================================================================
        torch.save(netG.state_dict(), gen_ckpt_path)
        torch.save(netD.state_dict(), disc_ckpt_path)
        torch.save({
            "epoch": epoch,
            "generator_state_dict": netG.state_dict(),
            "discriminator_state_dict": netD.state_dict(),
            "optimizerG_state_dict": optimizerG.state_dict(),
            "optimizerD_state_dict": optimizerD.state_dict(),
            "args": vars(args),
        }, full_ckpt_path)

    print(f"\n[Done] Training complete for {dataset} ({img_size}x{img_size})!")
    print(f"Sample grid saved to          : {os.path.join(args.output_dir, f'cgan_{dataset}_samples.png')}")
    print(f"Generator checkpoint saved to : {gen_ckpt_path}")
    print(f"Full checkpoint saved to      : {full_ckpt_path}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IF5152: Class-Conditional DCGAN (CIFAR-10 & LSUN)")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "lsun"],
                        help="Dataset to train on (cifar10: 32x32, lsun: 64x64)")
    parser.add_argument("--lsun-categories", type=str, default="bedroom,church_outdoor,kitchen",
                        help="Comma-separated list of LSUN categories when --dataset lsun")
    parser.add_argument("--data-dir", type=str, default=None, help="Root directory for dataset storage")
    parser.add_argument("--img-size", type=int, default=None, help="Image spatial resolution (32 or 64, default auto)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--latent-dim", type=int, default=100, help="Latent noise vector dimension")
    parser.add_argument("--embed-dim", type=int, default=100, help="Class label embedding dimension")
    parser.add_argument("--ngf", type=int, default=64, help="Generator feature maps multiplier")
    parser.add_argument("--ndf", type=int, default=64, help="Discriminator feature maps multiplier")
    parser.add_argument("--upsample-mode", type=str, default="nearest_conv", choices=["nearest_conv", "conv_transpose"],
                        help="Generator upsampling mode: 'nearest_conv' (anti-checkerboard) or 'conv_transpose' (Radford 2016)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam momentum beta1")
    parser.add_argument("--label-smoothing", type=float, default=0.9, help="One-sided label smoothing for real images")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda/mps/cpu)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_SAMPLES_DIR, help="Directory to save sample images")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="Directory to save checkpoints")
    parser.add_argument("--save-milestones", action="store_true", default=True, help="Save milestone snapshots")
    parser.add_argument("--quick-test", action="store_true", help="Quick run on subset for rapid verification")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint if present")
    parser.add_argument("--sample-only", action="store_true", help="Generate sample grid from saved checkpoint without training")
    args = parser.parse_args()

    train_cgan(args)
