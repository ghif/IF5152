"""
04e_resnet_acgan_cifar10.py
===========================
Modern ResNet Auxiliary Classifier Generative Adversarial Network (ResNet AC-GAN) on CIFAR-10.

Theoretical Foundations:
1. Canonical AC-GAN Formulation (Odena et al., ICML 2017 / arXiv:1610.09585):
   - Discriminator takes ONLY standard 3-channel RGB image x in R^{3 x 32 x 32} (NO label concatenation!).
   - Dual-headed output:
     a) Adversarial Source Head: D_adv(x) -> P(Source = real | x) in R (logits)
     b) Auxiliary Classifier Head: D_cls(x) -> P(Class = c | x) in R^{num_classes} (logits)
   - Multi-Task Objective:
     L_D = L_S(D_adv(x), y_source) + lambda_cls * L_C(D_cls(x), y_class)
     L_G = L_S(D_adv(G(z, c)), 1.0) + lambda_cls * L_C(D_cls(G(z, c)), c)

2. Modern Architectural Upgrades (Overcoming the DCGAN Bottleneck):
   - Nearest-Neighbor 2x Upsampling + 3x3 Convolutions in Generator:
     Completely eliminates transposed convolution (deconvolution) checkerboard artifacts.
   - Conditional Batch Normalization (CBN):
     Modulates feature activations directly across all residual stages using learned class embeddings.
   - Spectrally-Normalized ResNet Discriminator:
     Enforces 1-Lipschitz continuity on all convolutional blocks and linear heads, stabilizing the dual-task
     optimization game and preventing gradient vanishing or discriminator overconfidence.
   - Global Pooling Feature Extractor:
     Summarizes spatial representations psi(x) into a compact latent vector before routing to D_adv and D_cls.
   - Exponential Moving Average (EMA, beta=0.999):
     Smooths generator weights across training steps to yield high-fidelity, grain-free samples.
"""

import argparse
import copy
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
from tqdm import tqdm

from utils import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_SAMPLES_DIR,
    get_dataloader,
    get_device,
    save_image_grid,
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ==============================================================================
# 1. RESNET GENERATOR WITH CONDITIONAL BATCH NORMALIZATION (CBN)
# ==============================================================================
class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization (Dumoulin et al. 2017; Miyato & Koyama 2018).
    Applies standard batch normalization statistics, then modulates scale and shift
    dynamically as learned linear projections of the class embedding:
        h_norm = gamma(y) * ((h - mu) / sigma) + beta(y)
    """

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma_embed = nn.Embedding(num_classes, num_features)
        self.beta_embed = nn.Embedding(num_classes, num_features)

        # Initialize gamma around 1.0 and beta around 0.0
        nn.init.ones_(self.gamma_embed.weight)
        nn.init.zeros_(self.beta_embed.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        gamma = self.gamma_embed(y).view(-1, self.num_features, 1, 1)
        beta = self.beta_embed(y).view(-1, self.num_features, 1, 1)
        return gamma * out + beta


class ResBlockGen(nn.Module):
    """
    Residual Upsampling Block for Generator:
    - CBN(y) -> ReLU -> Nearest 2x Upsample -> Conv 3x3
    - CBN(y) -> ReLU -> Conv 3x3
    - Shortcut: Nearest 2x Upsample -> Conv 1x1
    Eliminates transposed convolution checkerboard artifacts.
    """

    def __init__(self, in_channels: int, out_channels: int, num_classes: int):
        super().__init__()
        self.cbn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.cbn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Shortcut branch: nearest upsample then 1x1 conv
        h_skip = F.interpolate(x, scale_factor=2, mode="nearest")
        h_skip = self.shortcut(h_skip)

        # Residual branch
        h = self.cbn1(x, y)
        h = F.relu(h)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)

        h = self.cbn2(h, y)
        h = F.relu(h)
        h = self.conv2(h)

        return h_skip + h


class ResNetACGANGenerator(nn.Module):
    """
    Modern Class-Conditional ResNet Generator for CIFAR-10 (3x32x32 RGB).
    Maps latent vector z ~ N(0, I) and class label c to a photorealistic image G(z, c).
    """

    def __init__(self, z_dim: int = 128, num_classes: int = 10, channels: int = 128):
        super().__init__()
        self.z_dim = z_dim
        self.channels = channels
        self.num_classes = num_classes

        # Linear projection: z -> (channels, 4, 4)
        self.fc = nn.Linear(z_dim, 4 * 4 * channels)

        # 3 Upsampling blocks: 4x4 -> 8x8 -> 16x16 -> 32x32
        self.block1 = ResBlockGen(channels, channels, num_classes)
        self.block2 = ResBlockGen(channels, channels, num_classes)
        self.block3 = ResBlockGen(channels, channels, num_classes)

        self.bn_out = nn.BatchNorm2d(channels)
        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, self.channels, 4, 4)

        h = self.block1(h, y)  # 4x4 -> 8x8
        h = self.block2(h, y)  # 8x8 -> 16x16
        h = self.block3(h, y)  # 16x16 -> 32x32

        h = F.relu(self.bn_out(h))
        out = torch.tanh(self.conv_out(h))
        return out


# ==============================================================================
# 2. DUAL-HEADED SPECTRALLY-NORMALIZED RESNET DISCRIMINATOR
# ==============================================================================
class ResBlockDiscFirst(nn.Module):
    """
    First Residual Block in Discriminator:
    Operates directly on raw image input (3x32x32 -> channels x 16x16).
    All convolutions constrained via Spectral Normalization.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 128):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.shortcut = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shortcut: 1x1 conv then 2x2 average pooling
        h_skip = self.shortcut(x)
        h_skip = F.avg_pool2d(h_skip, kernel_size=2)

        # Residual branch
        h = self.conv1(x)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.avg_pool2d(h, kernel_size=2)

        return h_skip + h


class ResBlockDisc(nn.Module):
    """
    Residual Downsampling Block for Discriminator with Spectral Normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        self.downsample = downsample
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        if in_channels != out_channels or downsample:
            self.shortcut = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shortcut branch
        h_skip = self.shortcut(x)
        if self.downsample:
            h_skip = F.avg_pool2d(h_skip, kernel_size=2)

        # Residual branch
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, kernel_size=2)

        return h_skip + h


class ResNetACGANDiscriminator(nn.Module):
    """
    Dual-Headed ResNet Discriminator for AC-GAN (Odena et al. 2017; Miyato et al. 2018).
    Takes ONLY 3x32x32 image x (no class label!), extracts global feature vector psi(x),
    and computes:
        1. D_adv(x): Adversarial source score (real vs. fake)
        2. D_cls(x): Multi-class auxiliary classification logits in R^{num_classes}
    """

    def __init__(self, num_classes: int = 10, channels: int = 128):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes

        # Downsampling blocks: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.block1 = ResBlockDiscFirst(3, channels)
        self.block2 = ResBlockDisc(channels, channels, downsample=True)
        self.block3 = ResBlockDisc(channels, channels, downsample=True)
        self.block4 = ResBlockDisc(channels, channels, downsample=False)

        # Dual Heads with Spectral Normalization
        self.fc_adv = spectral_norm(nn.Linear(channels, 1))
        self.fc_cls = spectral_norm(nn.Linear(channels, num_classes))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.block1(x)  # 32x32 -> 16x16
        h = self.block2(h)  # 16x16 -> 8x8
        h = self.block3(h)  # 8x8 -> 4x4
        h = self.block4(h)  # 4x4 -> 4x4
        h = F.relu(h)

        # Global sum pooling over spatial dimensions: (B, C, 4, 4) -> (B, C)
        psi = torch.sum(h, dim=(2, 3))

        # Head 1: Adversarial source logit
        adv_logits = self.fc_adv(psi).squeeze(1)

        # Head 2: Auxiliary class logits
        cls_logits = self.fc_cls(psi)

        return adv_logits, cls_logits


# ==============================================================================
# 3. EXPONENTIAL MOVING AVERAGE (EMA) TRACKER
# ==============================================================================
class EMAModel:
    """
    Exponential Moving Average (EMA) tracker for generator parameters.
    theta_EMA = decay * theta_EMA + (1 - decay) * theta_current
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = model
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    new_val = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                    self.shadow[name].copy_(new_val)

    def apply_shadow(self, model: nn.Module):
        """Context manager to temporarily swap model weights with smoothed EMA weights."""
        return _EMAShadowContext(model, self.shadow)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


class _EMAShadowContext:
    def __init__(self, model: nn.Module, shadow: dict):
        self.model = model
        self.shadow = shadow
        self.backup = {}

    def __enter__(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])


# ==============================================================================
# 4. TRAINING LOOP
# ==============================================================================
def train_resnet_acgan(
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.0002,
    z_dim: int = 128,
    gen_channels: int = 128,
    disc_channels: int = 128,
    loss_type: str = "bce",
    lambda_cls: float = 1.0,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    lr_decay: bool = True,
    device_str: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    samples_dir: Optional[str] = None,
    quick_test: bool = False,
    resume: bool = False,
) -> Tuple[ResNetACGANGenerator, ResNetACGANDiscriminator]:
    """
    Executes training loop for Modern ResNet AC-GAN on CIFAR-10.
    """
    device = get_device(device_str)
    print(f"[Device] Using {device} for ResNet AC-GAN CIFAR-10 training.")

    checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    samples_dir = samples_dir or DEFAULT_SAMPLES_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    milestones_dir = os.path.join(samples_dir, "resnet_acgan_cifar10_milestones")
    os.makedirs(milestones_dir, exist_ok=True)

    # Dedicated non-destructive filenames
    chk_path = os.path.join(checkpoint_dir, "resnet_acgan_cifar10_checkpoint.pt")
    gen_path = os.path.join(checkpoint_dir, "resnet_acgan_cifar10_generator.pt")
    gen_ema_path = os.path.join(checkpoint_dir, "resnet_acgan_cifar10_generator_ema.pt")
    disc_path = os.path.join(checkpoint_dir, "resnet_acgan_cifar10_discriminator.pt")

    num_classes = 10

    # 1. Instantiate networks
    netG = ResNetACGANGenerator(z_dim=z_dim, num_classes=num_classes, channels=gen_channels).to(device)
    netD = ResNetACGANDiscriminator(num_classes=num_classes, channels=disc_channels).to(device)

    ema_netG = EMAModel(netG, decay=ema_decay) if use_ema else None

    # 2. Loss criteria
    bce_criterion = nn.BCEWithLogitsLoss()
    cls_criterion = nn.CrossEntropyLoss()

    # 3. Optimizers: Adam with standard betas (0.0, 0.9) for spectral norm ResNets
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.0, 0.9))

    start_epoch = 1

    if resume and os.path.exists(chk_path):
        print(f"[Resume] Loading ResNet AC-GAN checkpoint from {chk_path}...")
        chk = torch.load(chk_path, map_location=device)
        netG.load_state_dict(chk["netG"])
        netD.load_state_dict(chk["netD"])
        optimizerG.load_state_dict(chk["optimizerG"])
        optimizerD.load_state_dict(chk["optimizerD"])
        for pg in optimizerG.param_groups:
            pg["lr"] = lr
            pg["initial_lr"] = lr
        for pg in optimizerD.param_groups:
            pg["lr"] = lr
            pg["initial_lr"] = lr
        if ema_netG is not None and "ema_netG" in chk:
            ema_netG.load_state_dict(chk["ema_netG"])
        start_epoch = chk["epoch"] + 1
        print(f"[Resume] Resuming at Epoch {start_epoch}/{epochs} (Previous D={chk.get('loss_D', 0):.4f}, G={chk.get('loss_G', 0):.4f})")

    # 4. Schedulers: Linear decay to 0 over remaining epochs
    total_remaining = max(1, epochs - start_epoch + 1)
    if lr_decay:
        def lr_lambda(epoch_idx: int):
            return max(0.0, 1.0 - (epoch_idx / total_remaining))
        schedulerD = optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lr_lambda)
        schedulerG = optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lr_lambda)
    else:
        schedulerD, schedulerG = None, None

    # 5. Fixed evaluation probe: 10 classes (rows) x 10 distinct noise vectors (columns) = 100 images
    fixed_classes = torch.arange(num_classes, device=device).repeat_interleave(10)
    fixed_z_base = torch.randn(10, z_dim, device=device)
    fixed_z = fixed_z_base.repeat(num_classes, 1)

    if quick_test:
        epochs = min(epochs, start_epoch)
        batch_size = min(batch_size, 16)

    # 6. Data loader
    dataloader = get_dataloader(
        dataset_name="cifar10",
        batch_size=batch_size,
        train=True,
        normalize_range=(-1.0, 1.0),
        quick_test=quick_test,
        max_samples=50 if quick_test else 50000,
    )

    print(f"\n========================================================")
    print(f"  Modern ResNet AC-GAN on CIFAR-10                       ")
    print(f"  Reference: Odena et al. (2017) + Miyato et al. (2018) ")
    print(f"========================================================")
    print(f"  Epochs           : {epochs} (starting at {start_epoch})")
    print(f"  Batch Size       : {batch_size}")
    print(f"  Latent Dim (z)   : {z_dim}")
    print(f"  Gen Channels     : {gen_channels}")
    print(f"  Disc Channels    : {disc_channels}")
    print(f"  Adversarial Loss : {loss_type.upper()}")
    print(f"  Classifier Weight: lambda_cls = {lambda_cls}")
    print(f"  EMA Tracking     : {'Active (decay=' + str(ema_decay) + ')' if use_ema else 'Disabled'}")
    print(f"  LR Decay         : {'Linear to 0' if lr_decay else 'Constant (' + str(lr) + ')'}")
    print(f"  Total Batches    : {len(dataloader)}")
    print(f"========================================================\n")

    for epoch in range(start_epoch, epochs + 1):
        netG.train()
        netD.train()
        running_d_loss = 0.0
        running_g_loss = 0.0
        running_real_acc = 0.0
        running_fake_acc = 0.0
        batch_count = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{epochs:02d}", leave=True)

        for real_imgs, real_labels in pbar:
            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)

            # Targets for BCE
            real_targets = torch.full((b_size,), 0.9, device=device)  # One-sided label smoothing
            fake_targets = torch.full((b_size,), 0.0, device=device)

            # ------------------------------------------------------------------
            # 1. TRAIN DISCRIMINATOR: max L_S + lambda_cls * L_C
            # ------------------------------------------------------------------
            optimizerD.zero_grad()

            # (a) Real images forward
            out_real_adv, out_real_cls = netD(real_imgs)
            loss_d_real_cls = cls_criterion(out_real_cls, real_labels)

            # (b) Fake images forward
            z = torch.randn(b_size, z_dim, device=device)
            fake_labels = torch.randint(0, num_classes, (b_size,), device=device)
            fake_imgs = netG(z, fake_labels)

            out_fake_adv, out_fake_cls = netD(fake_imgs.detach())
            loss_d_fake_cls = cls_criterion(out_fake_cls, fake_labels)

            # Adversarial source loss
            if loss_type == "hinge":
                loss_d_real_adv = F.relu(1.0 - out_real_adv).mean()
                loss_d_fake_adv = F.relu(1.0 + out_fake_adv).mean()
                loss_d_adv = loss_d_real_adv + loss_d_fake_adv
            else:  # Standard AC-GAN BCE
                loss_d_real_adv = bce_criterion(out_real_adv, real_targets)
                loss_d_fake_adv = bce_criterion(out_fake_adv, fake_targets)
                loss_d_adv = 0.5 * (loss_d_real_adv + loss_d_fake_adv)

            loss_d_cls = 0.5 * (loss_d_real_cls + loss_d_fake_cls)
            loss_d = loss_d_adv + lambda_cls * loss_d_cls

            loss_d.backward()
            optimizerD.step()

            # ------------------------------------------------------------------
            # 2. TRAIN GENERATOR: max lambda_cls * L_C - L_S
            # ------------------------------------------------------------------
            optimizerG.zero_grad()

            # Pass freshly generated images through updated D
            out_g_adv, out_g_cls = netD(fake_imgs)

            # Adversarial source loss for G
            if loss_type == "hinge":
                loss_g_adv = -out_g_adv.mean()
            else:
                loss_g_adv = bce_criterion(out_g_adv, torch.full((b_size,), 1.0, device=device))

            # Auxiliary classification loss for G
            loss_g_cls = cls_criterion(out_g_cls, fake_labels)

            loss_g = loss_g_adv + lambda_cls * loss_g_cls
            loss_g.backward()
            optimizerG.step()

            if ema_netG is not None:
                ema_netG.update(netG)

            # Calculate classification accuracies
            acc_real = (out_real_cls.argmax(dim=1) == real_labels).float().mean().item()
            acc_fake = (out_fake_cls.argmax(dim=1) == fake_labels).float().mean().item()

            running_d_loss += loss_d.item()
            running_g_loss += loss_g.item()
            running_real_acc += acc_real
            running_fake_acc += acc_fake
            batch_count += 1

            pbar.set_postfix({
                "Loss_D": f"{loss_d.item():.4f}",
                "Loss_G": f"{loss_g.item():.4f}",
                "Acc(real)": f"{acc_real * 100:.1f}%",
                "Acc(fake)": f"{acc_fake * 100:.1f}%",
            })

        if schedulerD is not None:
            schedulerD.step()
        if schedulerG is not None:
            schedulerG.step()

        avg_d = running_d_loss / max(1, batch_count)
        avg_g = running_g_loss / max(1, batch_count)
        avg_acc_real = (running_real_acc / max(1, batch_count)) * 100
        avg_acc_fake = (running_fake_acc / max(1, batch_count)) * 100
        print(f"[Epoch {epoch:02d}] Loss D: {avg_d:.4f} | Loss G: {avg_g:.4f} | Real Acc: {avg_acc_real:.1f}% | Fake Acc: {avg_acc_fake:.1f}%")

        # ------------------------------------------------------------------
        # 3. EVALUATION & SAMPLE SAVING
        # ------------------------------------------------------------------
        netG.eval()

        if ema_netG is not None:
            with ema_netG.apply_shadow(netG), torch.no_grad():
                sample_tensor_ema = netG(fixed_z, fixed_classes)

            ema_samples_path = os.path.join(samples_dir, "resnet_acgan_cifar10_samples_ema.png")
            save_image_grid(
                sample_tensor_ema,
                ema_samples_path,
                nrow=10,
                value_range=(-1.0, 1.0),
                title=f"ResNet AC-GAN CIFAR-10 Class Grid (EMA - Epoch {epoch})",
            )

            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                m_path = os.path.join(milestones_dir, f"epoch_{epoch:02d}.png")
                save_image_grid(
                    sample_tensor_ema,
                    m_path,
                    nrow=10,
                    value_range=(-1.0, 1.0),
                    title=f"ResNet AC-GAN CIFAR-10 (EMA - Epoch {epoch:02d})",
                )
                print(f"[Milestone] Saved snapshot (EMA): {m_path}")
        else:
            with torch.no_grad():
                sample_tensor_raw = netG(fixed_z, fixed_classes)
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                m_path = os.path.join(milestones_dir, f"epoch_{epoch:02d}.png")
                save_image_grid(
                    sample_tensor_raw,
                    m_path,
                    nrow=10,
                    value_range=(-1.0, 1.0),
                    title=f"ResNet AC-GAN CIFAR-10 (Epoch {epoch:02d})",
                )
                print(f"[Milestone] Saved snapshot: {m_path}")

        # Always save latest raw grid
        with torch.no_grad():
            sample_tensor_raw = netG(fixed_z, fixed_classes)
        raw_samples_path = os.path.join(samples_dir, "resnet_acgan_cifar10_samples.png")
        save_image_grid(
            sample_tensor_raw,
            raw_samples_path,
            nrow=10,
            value_range=(-1.0, 1.0),
            title=f"ResNet AC-GAN CIFAR-10 Class Grid (Raw - Epoch {epoch})",
        )

        # ------------------------------------------------------------------
        # 4. PERSIST CHECKPOINT (Non-Destructive)
        # ------------------------------------------------------------------
        chk_dict = {
            "epoch": epoch,
            "netG": netG.state_dict(),
            "netD": netD.state_dict(),
            "optimizerG": optimizerG.state_dict(),
            "optimizerD": optimizerD.state_dict(),
            "loss_D": avg_d,
            "loss_G": avg_g,
            "z_dim": z_dim,
            "gen_channels": gen_channels,
            "disc_channels": disc_channels,
            "num_classes": num_classes,
            "loss_type": loss_type,
        }
        if ema_netG is not None:
            chk_dict["ema_netG"] = ema_netG.state_dict()
            with ema_netG.apply_shadow(netG):
                torch.save(netG.state_dict(), gen_ema_path)

        torch.save(chk_dict, chk_path)
        torch.save(netG.state_dict(), gen_path)
        torch.save(netD.state_dict(), disc_path)

    print(f"\n[Training Complete] Final weights saved to:")
    print(f"  Generator (Raw): {gen_path}")
    if ema_netG is not None:
        print(f"  Generator (EMA): {gen_ema_path}")
    print(f"  Discriminator  : {disc_path}")
    print(f"  Full State     : {chk_path}")
    print(f"  Class Grid     : {raw_samples_path}\n")

    return netG, netD


def generate_grid_from_checkpoint(
    checkpoint_path: str,
    output_path: str = "samples/resnet_acgan_cifar10_samples.png",
    device_str: Optional[str] = None,
    z_dim: int = 128,
    gen_channels: int = 128,
    num_classes: int = 10,
    use_ema: bool = False,
) -> None:
    """
    Loads saved ResNet AC-GAN generator weights and outputs canonical 10x10 class matrix.
    """
    device = get_device(device_str)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ema_path = checkpoint_path.replace("generator.pt", "generator_ema.pt")
    if use_ema and os.path.exists(ema_path):
        print(f"[Eval Grid] Loading EMA weights from: {ema_path}")
        checkpoint_path = ema_path

    state = torch.load(checkpoint_path, map_location=device)
    raw_weights = state["netG"] if isinstance(state, dict) and "netG" in state else state

    netG = ResNetACGANGenerator(z_dim=z_dim, num_classes=num_classes, channels=gen_channels).to(device)
    netG.load_state_dict(raw_weights, strict=False)
    netG.eval()

    fixed_classes = torch.arange(num_classes, device=device).repeat_interleave(10)
    fixed_z_base = torch.randn(10, z_dim, device=device)
    fixed_z = fixed_z_base.repeat(num_classes, 1)

    with torch.no_grad():
        samples = netG(fixed_z, fixed_classes)

    save_image_grid(
        samples,
        output_path,
        nrow=10,
        value_range=(-1.0, 1.0),
        title="ResNet AC-GAN CIFAR-10 Evaluated Matrix",
    )
    print(f"[Generated Grid] Saved 10x10 class matrix to {output_path}")


# ==============================================================================
# 5. CLI ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="IF5152: Modern ResNet Auxiliary Classifier GAN (ResNet AC-GAN) on CIFAR-10"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate (default: 0.0002)")
    parser.add_argument("--z-dim", type=int, default=128, help="Latent noise vector dimension (default: 128)")
    parser.add_argument("--gen-channels", type=int, default=128, help="Generator base channel count (default: 128)")
    parser.add_argument("--disc-channels", type=int, default=128, help="Discriminator base channel count (default: 128)")
    parser.add_argument("--loss-type", type=str, default="bce", choices=["bce", "hinge"], help="Adversarial loss formulation (default: bce)")
    parser.add_argument("--lambda-cls", type=float, default=1.0, help="Auxiliary classification loss weight (default: 1.0)")
    parser.add_argument("--no-ema", action="store_true", help="Disable Exponential Moving Average tracking for Generator")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay rate (default: 0.999)")
    parser.add_argument("--no-lr-decay", action="store_true", help="Disable linear learning rate decay")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--quick-test", action="store_true", help="Quick smoke test (1 epoch, 50 samples)")
    parser.add_argument("--eval-grid", action="store_true", help="Generate 10x10 class grid from saved checkpoint")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights for evaluation grid")
    args = parser.parse_args()

    if args.eval_grid:
        chk = os.path.join(DEFAULT_CHECKPOINT_DIR, "resnet_acgan_cifar10_generator.pt")
        out = os.path.join(DEFAULT_SAMPLES_DIR, "resnet_acgan_cifar10_samples.png")
        generate_grid_from_checkpoint(
            chk,
            out,
            device_str=args.device,
            z_dim=args.z_dim,
            gen_channels=args.gen_channels,
            use_ema=args.use_ema,
        )
        return

    train_resnet_acgan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        z_dim=args.z_dim,
        gen_channels=args.gen_channels,
        disc_channels=args.disc_channels,
        loss_type=args.loss_type,
        lambda_cls=args.lambda_cls,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        lr_decay=not args.no_lr_decay,
        device_str=args.device,
        quick_test=args.quick_test,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
