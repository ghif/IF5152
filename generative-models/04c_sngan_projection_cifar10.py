"""
04c_sngan_projection_cifar10.py
===============================
Class-Conditional ResNet Generative Adversarial Network with:
1. Spectral Normalization (Miyato et al., ICLR 2018)
2. Conditional Batch Normalization (Dumoulin et al. 2017)
3. Projection Discriminator (Miyato & Koyama, ICLR 2018)
4. Hinge Adversarial Loss (Miyato et al., 2018; Brock et al., 2019)

Benchmark on CIFAR-10 (3x32x32 RGB):
- Vanilla DCGAN: Inception Score ~ 6.5, FID ~ 38
- ResNet SNGAN:  Inception Score ~ 8.4 - 8.6, FID ~ 14 - 17

Key Architectural Principles:
1. Generator:
   - Eliminates transposed convolution checkerboard artifacts by using
     nearest-neighbor upsampling (2x) followed by 3x3 convolutions.
   - Modulates affine batch-norm parameters conditioned on class label y:
     CBN(h | y) = gamma(y) * ((h - mu) / sigma) + beta(y)
2. Discriminator:
   - Spectral Normalization on ALL conv and linear layers enforces 1-Lipschitz
     continuity, preventing gradient vanishing or explosion on disjoint manifolds.
   - Projection Discriminator decomposes the prediction into:
     D(x, y) = phi(psi(x)) + <psi(x), v_y>
     where phi(psi(x)) is unconditional realism and <psi(x), v_y> is label alignment.
3. Hinge Loss:
   - L_D = E[max(0, 1 - D(x, y))] + E[max(0, 1 + D(G(z, y), y))]
   - L_G = -E[D(G(z, y), y)]
"""

import argparse
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
# 1. CONDITIONAL BATCH NORMALIZATION (CBN)
# ==============================================================================
class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization (Dumoulin et al., ICLR 2017).
    Standardizes activation statistics spatially, then modulates the affine
    scale gamma and shift beta as learned functions of class label y.
    """

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed_gamma = nn.Embedding(num_classes, num_features)
        self.embed_beta = nn.Embedding(num_classes, num_features)

        # Initialize gamma around 1.0 and beta around 0.0
        nn.init.ones_(self.embed_gamma.weight)
        nn.init.zeros_(self.embed_beta.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        gamma = self.embed_gamma(y).unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        beta = self.embed_beta(y).unsqueeze(2).unsqueeze(3)    # (B, C, 1, 1)
        return gamma * out + beta


# ==============================================================================
# 2. RESNET GENERATOR BLOCKS
# ==============================================================================
class ResBlockGen(nn.Module):
    """
    Residual Upsampling Block for Generator:
    - CBN(y) -> ReLU -> Nearest 2x Upsample -> Conv 3x3
    - CBN(y) -> ReLU -> Conv 3x3
    - Shortcut: Nearest 2x Upsample -> Conv 1x1 (if channel mismatch or upsample)
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


class ResNetGenerator(nn.Module):
    """
    Class-Conditional ResNet Generator for CIFAR-10 (3x32x32 RGB).
    Maps noise z ~ N(0, I) and class y to 3x32x32 image in range [-1, 1].
    """

    def __init__(self, z_dim: int = 128, num_classes: int = 10, channels: int = 256):
        super().__init__()
        self.z_dim = z_dim
        self.channels = channels

        # Linear projection from latent space to 4x4 spatial feature map
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
# 3. RESNET DISCRIMINATOR BLOCKS (WITH SPECTRAL NORMALIZATION)
# ==============================================================================
class ResBlockDiscFirst(nn.Module):
    """
    First Residual Block in Discriminator:
    Operates directly on raw image input without initial ReLU/normalization.
    Downsamples 32x32 -> 16x16.
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
    Standard Residual Downsampling / Preserving Block for Discriminator.
    All convolutions are constrained via Spectral Normalization.
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


class ResNetProjectionDiscriminator(nn.Module):
    """
    ResNet Discriminator with Projection Conditioning (Miyato & Koyama, ICLR 2018).
    Extracts deep global visual representation psi(x) in R^channels, then computes:
        D(x, y) = phi(psi(x)) + <psi(x), v_y>
    """

    def __init__(self, num_classes: int = 10, channels: int = 128):
        super().__init__()
        self.channels = channels

        # Downsampling blocks: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.block1 = ResBlockDiscFirst(3, channels)
        self.block2 = ResBlockDisc(channels, channels, downsample=True)
        self.block3 = ResBlockDisc(channels, channels, downsample=True)
        self.block4 = ResBlockDisc(channels, channels, downsample=False)

        # Unconditional realism linear scoring head
        self.fc = spectral_norm(nn.Linear(channels, 1))

        # Learned class embedding projection vector
        self.embed = spectral_norm(nn.Embedding(num_classes, channels))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)  # 32x32 -> 16x16
        h = self.block2(h)  # 16x16 -> 8x8
        h = self.block3(h)  # 8x8 -> 4x4
        h = self.block4(h)  # 4x4 -> 4x4
        h = F.relu(h)

        # Global sum pooling over spatial dimensions: (B, C, 4, 4) -> (B, C)
        psi = torch.sum(h, dim=(2, 3))

        # 1. Unconditional score: "Does this look like a realistic image?"
        uncond_score = self.fc(psi).squeeze(1)

        # 2. Projection score: "Does this image match class y?"
        class_v = self.embed(y)  # (B, C)
        cond_projection = torch.sum(psi * class_v, dim=1)  # Inner product

        # Total discriminator logit
        return uncond_score + cond_projection


# ==============================================================================
# 4. EXPONENTIAL MOVING AVERAGE (EMA) FOR GENERATOR WEIGHTS
# ==============================================================================
class _EMAShadowContext:
    def __init__(self, model: nn.Module, shadow: dict):
        self.model = model
        self.shadow = shadow
        self.backup = {}

    def __enter__(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone().detach()
                param.data.copy_(self.shadow[name])
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])


class EMAModel:
    """
    Maintains an Exponential Moving Average (EMA) of model parameters:
        theta_ema <- decay * theta_ema + (1 - decay) * theta
    Used extensively in modern GANs (SNGAN, BigGAN, StyleGAN) to produce
    significantly sharper, less noisy evaluation samples than raw stochastic weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow = {k: v.clone().detach() for k, v in state_dict.items()}

    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def apply_shadow(self, model: nn.Module):
        """Context manager to temporarily apply shadow weights to model."""
        return _EMAShadowContext(model, self.shadow)


# ==============================================================================
# 5. TRAINING PIPELINE WITH HINGE LOSS
# ==============================================================================
def train_sngan(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.0002,
    z_dim: int = 128,
    num_classes: int = 10,
    gen_channels: int = 128,
    disc_channels: int = 128,
    n_dis: int = 1,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    lr_decay: bool = False,
    device_str: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    samples_dir: Optional[str] = None,
    quick_test: bool = False,
    resume: bool = False,
) -> Tuple[ResNetGenerator, ResNetProjectionDiscriminator]:
    """
    Executes training loop for ResNet SNGAN with Projection Discriminator.
    Supports resuming, EMA generator tracking, learning rate decay, and n_dis critic ratios.
    """
    device = get_device(device_str)
    print(f"[Device] Using {device} for ResNet SNGAN CIFAR-10 training.")

    checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    samples_dir = samples_dir or DEFAULT_SAMPLES_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    milestones_dir = os.path.join(samples_dir, "sngan_cifar10_milestones")
    os.makedirs(milestones_dir, exist_ok=True)

    chk_path = os.path.join(checkpoint_dir, "sngan_cifar10_checkpoint.pt")
    gen_path = os.path.join(checkpoint_dir, "sngan_cifar10_generator.pt")
    gen_ema_path = os.path.join(checkpoint_dir, "sngan_cifar10_generator_ema.pt")
    disc_path = os.path.join(checkpoint_dir, "sngan_cifar10_discriminator.pt")

    start_epoch = 1
    loaded_chk = None

    # Auto-detect channels from existing checkpoint if resuming
    if resume and os.path.exists(chk_path):
        print(f"[Resume] Inspecting existing checkpoint: {chk_path}...")
        loaded_chk = torch.load(chk_path, map_location=device)
        if "gen_channels" in loaded_chk:
            gen_channels = loaded_chk["gen_channels"]
        elif "netG" in loaded_chk and "fc.weight" in loaded_chk["netG"]:
            gen_channels = loaded_chk["netG"]["fc.weight"].shape[0] // 16

        if "disc_channels" in loaded_chk:
            disc_channels = loaded_chk["disc_channels"]
        elif "netD" in loaded_chk:
            for k in ["fc.weight_orig", "fc.weight"]:
                if k in loaded_chk["netD"]:
                    disc_channels = loaded_chk["netD"][k].shape[1]
                    break
        print(f"[Resume] Using architecture channels: Generator={gen_channels}, Discriminator={disc_channels}")

    # 1. Instantiate networks
    netG = ResNetGenerator(z_dim=z_dim, num_classes=num_classes, channels=gen_channels).to(device)
    netD = ResNetProjectionDiscriminator(num_classes=num_classes, channels=disc_channels).to(device)

    # 2. EMA Tracker
    ema_netG = EMAModel(netG, decay=ema_decay) if use_ema else None

    # 3. Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.0, 0.9))

    # Load weights and optimizer states if resuming
    if loaded_chk is not None:
        netG.load_state_dict(loaded_chk["netG"])
        netD.load_state_dict(loaded_chk["netD"])
        if "optimizerG" in loaded_chk:
            optimizerG.load_state_dict(loaded_chk["optimizerG"])
        if "optimizerD" in loaded_chk:
            optimizerD.load_state_dict(loaded_chk["optimizerD"])
        if ema_netG is not None and "ema_netG" in loaded_chk:
            ema_netG.load_state_dict(loaded_chk["ema_netG"])
            print("[Resume] Loaded EMA generator shadow weights.")
        elif ema_netG is not None:
            print("[Resume] Initialized EMA weights from resumed Generator.")
        start_epoch = loaded_chk["epoch"] + 1
        print(f"[Resume] Successfully resumed at Epoch {start_epoch}/{epochs} (Previous D={loaded_chk.get('loss_D', 0):.4f}, G={loaded_chk.get('loss_G', 0):.4f})")

    # 4. Learning rate schedulers
    total_remaining_epochs = max(1, epochs - start_epoch + 1)
    if lr_decay:
        def lr_lambda(epoch_idx: int):
            return max(0.0, 1.0 - (epoch_idx / total_remaining_epochs))
        schedulerD = optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lr_lambda)
        schedulerG = optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lr_lambda)
        print(f"[LR Scheduler] Linear decay active from Epoch {start_epoch} to {epochs}.")
    else:
        schedulerD, schedulerG = None, None

    # 5. Fixed probe for canonical 10x10 matrix evaluation
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
    print(f"  ResNet SNGAN with Projection Discriminator (CIFAR-10) ")
    print(f"========================================================")
    print(f"  Epochs           : {epochs} (starting at {start_epoch})")
    print(f"  Batch Size       : {batch_size}")
    print(f"  Critic Steps     : n_dis = {n_dis}")
    print(f"  EMA Tracking     : {'Active (decay=' + str(ema_decay) + ')' if use_ema else 'Disabled'}")
    print(f"  LR Decay         : {'Linear to 0' if lr_decay else 'Constant (' + str(lr) + ')'}")
    print(f"  Generator Ch     : {gen_channels} (ResNet + CBN + Nearest 2x)")
    print(f"  Discriminator Ch : {disc_channels} (ResNet + SN + Projection)")
    print(f"  Loss Function    : Hinge Loss")
    print(f"  Total Batches    : {len(dataloader)}")
    print(f"========================================================\n")

    global_step = 0

    for epoch in range(start_epoch, epochs + 1):
        netG.train()
        netD.train()
        running_d_loss = 0.0
        running_g_loss = 0.0
        d_count = 0
        g_count = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{epochs:02d}", leave=True)

        for real_imgs, labels in pbar:
            global_step += 1
            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)

            # ------------------------------------------------------------------
            # 1. TRAIN DISCRIMINATOR with HINGE LOSS
            # ------------------------------------------------------------------
            optimizerD.zero_grad()

            d_real = netD(real_imgs, labels)
            loss_d_real = torch.mean(F.relu(1.0 - d_real))

            z = torch.randn(b_size, z_dim, device=device)
            fake_labels = torch.randint(0, num_classes, (b_size,), device=device)
            fake_imgs = netG(z, fake_labels)

            d_fake = netD(fake_imgs.detach(), fake_labels)
            loss_d_fake = torch.mean(F.relu(1.0 + d_fake))

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizerD.step()

            running_d_loss += loss_d.item()
            d_count += 1

            # ------------------------------------------------------------------
            # 2. TRAIN GENERATOR with HINGE LOSS (every n_dis steps)
            # ------------------------------------------------------------------
            if global_step % n_dis == 0:
                optimizerG.zero_grad()

                z = torch.randn(b_size, z_dim, device=device)
                gen_labels = torch.randint(0, num_classes, (b_size,), device=device)
                fake_imgs = netG(z, gen_labels)

                d_fake_for_g = netD(fake_imgs, gen_labels)
                loss_g = -torch.mean(d_fake_for_g)

                loss_g.backward()
                optimizerG.step()

                if ema_netG is not None:
                    ema_netG.update(netG)

                running_g_loss += loss_g.item()
                g_count += 1

            cur_lr = optimizerG.param_groups[0]["lr"]
            pbar.set_postfix({
                "Loss_D": f"{loss_d.item():.4f}",
                "Loss_G": f"{(running_g_loss / max(1, g_count)):.4f}",
                "D(real)": f"{d_real.mean().item():+.2f}",
                "D(fake)": f"{d_fake.mean().item():+.2f}",
                "LR": f"{cur_lr:.1e}",
            })

        if schedulerD is not None:
            schedulerD.step()
        if schedulerG is not None:
            schedulerG.step()

        avg_d = running_d_loss / max(1, d_count)
        avg_g = running_g_loss / max(1, g_count)
        print(f"[Epoch {epoch:02d}] Average Loss D: {avg_d:.4f} | Average Loss G: {avg_g:.4f}")

        # ------------------------------------------------------------------
        # 3. EVALUATION & SAMPLE SAVING (EMA and Raw)
        # ------------------------------------------------------------------
        netG.eval()

        # Generate sample grid with EMA if enabled
        if ema_netG is not None:
            with ema_netG.apply_shadow(netG), torch.no_grad():
                sample_tensor_ema = netG(fixed_z, fixed_classes)

            ema_samples_path = os.path.join(samples_dir, "sngan_cifar10_samples_ema.png")
            save_image_grid(
                sample_tensor_ema,
                ema_samples_path,
                nrow=10,
                value_range=(-1.0, 1.0),
                title=f"SNGAN CIFAR-10 Class Grid (EMA - Epoch {epoch})",
            )

            # Milestones prefer EMA weights for clean presentation
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                m_path = os.path.join(milestones_dir, f"epoch_{epoch:02d}.png")
                save_image_grid(
                    sample_tensor_ema,
                    m_path,
                    nrow=10,
                    value_range=(-1.0, 1.0),
                    title=f"ResNet SNGAN CIFAR-10 (EMA - Epoch {epoch:02d})",
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
                    title=f"ResNet SNGAN CIFAR-10 (Epoch {epoch:02d})",
                )
                print(f"[Milestone] Saved snapshot: {m_path}")

        # Always save latest raw grid
        with torch.no_grad():
            sample_tensor_raw = netG(fixed_z, fixed_classes)
        raw_samples_path = os.path.join(samples_dir, "sngan_cifar10_samples.png")
        save_image_grid(
            sample_tensor_raw,
            raw_samples_path,
            nrow=10,
            value_range=(-1.0, 1.0),
            title=f"SNGAN CIFAR-10 Class Grid (Raw - Epoch {epoch})",
        )

        # ------------------------------------------------------------------
        # 4. PERSIST CHECKPOINT
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
            "num_classes": num_classes,
            "gen_channels": gen_channels,
            "disc_channels": disc_channels,
        }
        if ema_netG is not None:
            chk_dict["ema_netG"] = ema_netG.state_dict()
            torch.save(ema_netG.shadow, gen_ema_path)

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
    output_path: str = "samples/sngan_cifar10_samples.png",
    device_str: Optional[str] = None,
    z_dim: int = 128,
    num_classes: int = 10,
    use_ema: bool = False,
) -> None:
    """
    Loads saved generator weights and outputs canonical 10x10 class matrix.
    Auto-detects channel width and supports EMA weights.
    """
    device = get_device(device_str)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Check for EMA weights if requested
    ema_path = checkpoint_path.replace("generator.pt", "generator_ema.pt")
    if use_ema and os.path.exists(ema_path):
        print(f"[Eval Grid] Loading EMA weights from: {ema_path}")
        checkpoint_path = ema_path

    state = torch.load(checkpoint_path, map_location=device)
    raw_weights = state["netG"] if "netG" in state else state

    # Auto-detect channels
    channels = 128
    if "fc.weight" in raw_weights:
        channels = raw_weights["fc.weight"].shape[0] // 16
    print(f"[Eval Grid] Auto-detected Generator channels: {channels}")

    netG = ResNetGenerator(z_dim=z_dim, num_classes=num_classes, channels=channels).to(device)
    netG.load_state_dict(raw_weights)
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
        title="ResNet SNGAN CIFAR-10 Evaluated Matrix",
    )
    print(f"[Generated Grid] Saved 10x10 class matrix to {output_path}")


# ==============================================================================
# 6. CLI ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="IF5152: ResNet SNGAN with Projection Discriminator on CIFAR-10"
    )
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs (default: 80)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate (default: 0.0002)")
    parser.add_argument("--z-dim", type=int, default=128, help="Latent noise vector dimension (default: 128)")
    parser.add_argument("--gen-channels", type=int, default=128, help="Generator base feature channels (default: 128)")
    parser.add_argument("--disc-channels", type=int, default=128, help="Discriminator base feature channels (default: 128)")
    parser.add_argument("--n-dis", type=int, default=1, help="Discriminator update steps per Generator update (default: 1)")
    parser.add_argument("--no-ema", action="store_true", help="Disable Exponential Moving Average tracking for Generator")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay rate (default: 0.999)")
    parser.add_argument("--lr-decay", action="store_true", help="Linearly decay learning rate to 0 over remaining epochs")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--quick-test", action="store_true", help="Quick smoke test (1 epoch, 50 samples)")
    parser.add_argument("--eval-grid", action="store_true", help="Generate 10x10 class grid from saved checkpoint")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights for evaluation grid")
    args = parser.parse_args()

    if args.eval_grid:
        chk = os.path.join(DEFAULT_CHECKPOINT_DIR, "sngan_cifar10_generator.pt")
        out = os.path.join(DEFAULT_SAMPLES_DIR, "sngan_cifar10_samples.png")
        generate_grid_from_checkpoint(
            chk,
            out,
            device_str=args.device,
            z_dim=args.z_dim,
            use_ema=args.use_ema,
        )
        return

    train_sngan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        z_dim=args.z_dim,
        gen_channels=args.gen_channels,
        disc_channels=args.disc_channels,
        n_dis=args.n_dis,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        lr_decay=args.lr_decay,
        device_str=args.device,
        quick_test=args.quick_test,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
