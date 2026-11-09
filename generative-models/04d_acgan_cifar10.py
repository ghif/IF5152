"""
04d_acgan_cifar10.py
====================
Auxiliary Classifier Generative Adversarial Network (AC-GAN) on CIFAR-10.
Reference:
  Augustus Odena, Christopher Olah, Jonathon Shlens (ICML 2017 / arXiv:1610.09585)
  "Conditional Image Synthesis With Auxiliary Classifier GANs"
  https://arxiv.org/abs/1610.09585

Key Architectural & Algorithmic Principles:
1. Dual-Headed Discriminator:
   - Takes ONLY standard 3-channel RGB image x in R^{3 x 32 x 32} (NO class label input!).
   - Decomposes into two dedicated output heads:
     a) Adversarial Source Head: P(Source = real | x) in [0, 1]
     b) Auxiliary Classifier Head: P(Class = c | x) in R^{num_classes}
2. Multi-Task Composite Loss:
   - L_S = E[log P(S=real | x_real)] + E[log P(S=fake | G(z, c))]   (Adversarial BCE)
   - L_C = E[log P(C=c | x_real)] + E[log P(C=c | G(z, c))]         (Classification CrossEntropy)
   - Discriminator trains to maximize: L_S + L_C
   - Generator trains to maximize: L_C - L_S
3. Representation Disentanglement:
   - Forcing the discriminator to predict class labels c for both real and generated
     images prevents mode collapse and encourages D to build rich semantic representations.
4. Modern High-Quality Enhancements:
   - Exponential Moving Average (EMA, beta=0.999) of Generator weights for crisp sampling.
   - Linear learning rate decay.
   - Classification accuracy tracking for both real data and synthesized fakes.
"""

import argparse
import os
import sys
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


def weights_init_acgan(m: nn.Module) -> None:
    """
    Standard DCGAN / AC-GAN weight initialization (Radford et al. 2016; Odena et al. 2016).
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Embedding") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ==============================================================================
# 1. AC-GAN GENERATOR (Odena et al. 2016)
# ==============================================================================
class ACGANGenerator(nn.Module):
    """
    AC-GAN Generator for 3x32x32 RGB CIFAR-10 images.
    Maps latent noise vector z and class label c to a photorealistic image G(z, c).
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

        # Class embedding
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        in_dim = latent_dim + embed_dim

        # Transposed convolution upsampling stages: 1x1 -> 4x4 -> 8x8 -> 16x16 -> 32x32
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 4),
            nn.ReLU(True),
        )

        self.conv_blocks = nn.Sequential(
            # (ngf * 4) x 4 x 4 -> (ngf * 2) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf * 2) x 8 x 8 -> ngf x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # ngf x 16 x 16 -> channels x 32 x 32
            nn.ConvTranspose2d(ngf, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

        self.ngf = ngf
        self.apply(weights_init_acgan)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if z.dim() == 4:
            z = z.flatten(1)
        c_emb = self.label_emb(c)
        zc = torch.cat([z, c_emb], dim=1)
        h = self.fc(zc)
        h = h.view(-1, self.ngf * 4, 4, 4)
        out = self.conv_blocks(h)
        return out


# ==============================================================================
# 2. AC-GAN DISCRIMINATOR WITH AUXILIARY CLASSIFIER (Odena et al. 2016)
# ==============================================================================
class ACGANDiscriminator(nn.Module):
    """
    AC-GAN Discriminator for 3x32x32 RGB CIFAR-10 images.
    Input: Standard 3-channel RGB image (NO class label concatenated!).
    Outputs:
      1. adv_out: Scalar logit for source validity (real vs. fake).
      2. cls_out: Vector of 10 class logits for auxiliary class prediction.
    """

    def __init__(
        self,
        num_classes: int = 10,
        ndf: int = 64,
        channels: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Shared convolutional feature extractor
        self.features = nn.Sequential(
            # 3 x 32 x 32 -> ndf x 16 x 16
            nn.Conv2d(channels, ndf, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),

            # ndf x 16 x 16 -> (ndf * 2) x 8 x 8
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),

            # (ndf * 2) x 8 x 8 -> (ndf * 2) x 8 x 8 (strided) -> (ndf * 4) x 4 x 4
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),

            # (ndf * 4) x 4 x 4 -> (ndf * 4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),

            # (ndf * 4) x 4 x 4 -> (ndf * 8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),

            # Global average pooling: (ndf * 8) x 4 x 4 -> (ndf * 8) x 1 x 1
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        flat_dim = ndf * 8

        # Head 1: Adversarial discrimination (real vs. fake source)
        self.adv_head = nn.Sequential(
            nn.Linear(flat_dim, 1),
        )

        # Head 2: Auxiliary classifier (predict class c)
        self.cls_head = nn.Sequential(
            nn.Linear(flat_dim, num_classes),
        )

        self.apply(weights_init_acgan)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        feat_flat = feat.view(feat.size(0), -1)

        adv_logits = self.adv_head(feat_flat).squeeze(1)  # (B,)
        cls_logits = self.cls_head(feat_flat)             # (B, num_classes)

        return adv_logits, cls_logits


# ==============================================================================
# 3. EXPONENTIAL MOVING AVERAGE (EMA) HELPER
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
    Maintains an Exponential Moving Average (EMA) of Generator parameters:
        theta_ema <- decay * theta_ema + (1 - decay) * theta
    Significantly enhances image sharpness and suppresses stochastic sample noise.
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

    def apply_shadow(self, model: nn.Module):
        return _EMAShadowContext(model, self.shadow)


# ==============================================================================
# 4. TRAINING PIPELINE (AC-GAN MULTI-TASK OBJECTIVE)
# ==============================================================================
def train_acgan(
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.0002,
    latent_dim: int = 100,
    embed_dim: int = 100,
    num_classes: int = 10,
    ngf: int = 64,
    ndf: int = 64,
    lambda_cls: float = 1.0,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    lr_decay: bool = True,
    device_str: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    samples_dir: Optional[str] = None,
    quick_test: bool = False,
    resume: bool = False,
) -> Tuple[ACGANGenerator, ACGANDiscriminator]:
    """
    Executes training loop for AC-GAN on CIFAR-10.
    """
    device = get_device(device_str)
    print(f"[Device] Using {device} for AC-GAN CIFAR-10 training.")

    checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    samples_dir = samples_dir or DEFAULT_SAMPLES_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    milestones_dir = os.path.join(samples_dir, "acgan_cifar10_milestones")
    os.makedirs(milestones_dir, exist_ok=True)

    chk_path = os.path.join(checkpoint_dir, "acgan_cifar10_checkpoint.pt")
    gen_path = os.path.join(checkpoint_dir, "acgan_cifar10_generator.pt")
    gen_ema_path = os.path.join(checkpoint_dir, "acgan_cifar10_generator_ema.pt")
    disc_path = os.path.join(checkpoint_dir, "acgan_cifar10_discriminator.pt")

    # 1. Instantiate networks
    netG = ACGANGenerator(latent_dim=latent_dim, embed_dim=embed_dim, num_classes=num_classes, ngf=ngf).to(device)
    netD = ACGANDiscriminator(num_classes=num_classes, ndf=ndf).to(device)

    ema_netG = EMAModel(netG, decay=ema_decay) if use_ema else None

    # 2. Losses
    adversarial_criterion = nn.BCEWithLogitsLoss()
    classification_criterion = nn.CrossEntropyLoss()

    # 3. Optimizers: Adam with standard DCGAN / AC-GAN hyperparameters
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    start_epoch = 1

    if resume and os.path.exists(chk_path):
        print(f"[Resume] Loading AC-GAN checkpoint from {chk_path}...")
        chk = torch.load(chk_path, map_location=device)
        netG.load_state_dict(chk["netG"])
        netD.load_state_dict(chk["netD"])
        optimizerG.load_state_dict(chk["optimizerG"])
        optimizerD.load_state_dict(chk["optimizerD"])
        # Reset learning rates to target lr so decayed schedulers from previous runs do not stall
        for param_group in optimizerG.param_groups:
            param_group["lr"] = lr
            param_group["initial_lr"] = lr
        for param_group in optimizerD.param_groups:
            param_group["lr"] = lr
            param_group["initial_lr"] = lr
        if ema_netG is not None and "ema_netG" in chk:
            ema_netG.load_state_dict(chk["ema_netG"])
        start_epoch = chk["epoch"] + 1
        print(f"[Resume] Resuming at Epoch {start_epoch}/{epochs} (Previous D={chk.get('loss_D', 0):.4f}, G={chk.get('loss_G', 0):.4f})")

    # 4. Schedulers
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
    fixed_z_base = torch.randn(10, latent_dim, device=device)
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
    print(f"  Auxiliary Classifier GAN (AC-GAN) on CIFAR-10         ")
    print(f"  Reference: Odena et al. (ICML 2017 / arXiv:1610.09585)")
    print(f"========================================================")
    print(f"  Epochs           : {epochs} (starting at {start_epoch})")
    print(f"  Batch Size       : {batch_size}")
    print(f"  Latent Dim (z)   : {latent_dim}")
    print(f"  Class Embed (c)  : {embed_dim}")
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

            # Targets with one-sided label smoothing for real images (0.9)
            real_targets = torch.full((b_size,), 0.9, device=device)
            fake_targets = torch.full((b_size,), 0.0, device=device)

            # ------------------------------------------------------------------
            # 1. TRAIN DISCRIMINATOR: max L_S + L_C
            # ------------------------------------------------------------------
            optimizerD.zero_grad()

            # (a) Real images forward
            out_real_adv, out_real_cls = netD(real_imgs)
            loss_d_real_adv = adversarial_criterion(out_real_adv, real_targets)
            loss_d_real_cls = classification_criterion(out_real_cls, real_labels)

            # (b) Fake images forward
            z = torch.randn(b_size, latent_dim, device=device)
            fake_labels = torch.randint(0, num_classes, (b_size,), device=device)
            fake_imgs = netG(z, fake_labels)

            out_fake_adv, out_fake_cls = netD(fake_imgs.detach())
            loss_d_fake_adv = adversarial_criterion(out_fake_adv, fake_targets)
            loss_d_fake_cls = classification_criterion(out_fake_cls, fake_labels)

            # Composite Discriminator Loss: L_S + L_C
            loss_d_adv = 0.5 * (loss_d_real_adv + loss_d_fake_adv)
            loss_d_cls = 0.5 * (loss_d_real_cls + loss_d_fake_cls)
            loss_d = loss_d_adv + lambda_cls * loss_d_cls

            loss_d.backward()
            optimizerD.step()

            # ------------------------------------------------------------------
            # 2. TRAIN GENERATOR: max L_C - L_S
            # ------------------------------------------------------------------
            optimizerG.zero_grad()

            # Pass freshly generated images through updated D
            out_g_adv, out_g_cls = netD(fake_imgs)
            # Generator wants D to classify fakes as REAL (targets = 1.0)
            loss_g_adv = adversarial_criterion(out_g_adv, torch.full((b_size,), 1.0, device=device))
            # Generator wants D auxiliary classifier to correctly recognize intended class
            loss_g_cls = classification_criterion(out_g_cls, fake_labels)

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

            ema_samples_path = os.path.join(samples_dir, "acgan_cifar10_samples_ema.png")
            save_image_grid(
                sample_tensor_ema,
                ema_samples_path,
                nrow=10,
                value_range=(-1.0, 1.0),
                title=f"AC-GAN CIFAR-10 Class Grid (EMA - Epoch {epoch})",
            )

            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                m_path = os.path.join(milestones_dir, f"epoch_{epoch:02d}.png")
                save_image_grid(
                    sample_tensor_ema,
                    m_path,
                    nrow=10,
                    value_range=(-1.0, 1.0),
                    title=f"AC-GAN CIFAR-10 (EMA - Epoch {epoch:02d})",
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
                    title=f"AC-GAN CIFAR-10 (Epoch {epoch:02d})",
                )
                print(f"[Milestone] Saved snapshot: {m_path}")

        # Always save latest raw grid
        with torch.no_grad():
            sample_tensor_raw = netG(fixed_z, fixed_classes)
        raw_samples_path = os.path.join(samples_dir, "acgan_cifar10_samples.png")
        save_image_grid(
            sample_tensor_raw,
            raw_samples_path,
            nrow=10,
            value_range=(-1.0, 1.0),
            title=f"AC-GAN CIFAR-10 Class Grid (Raw - Epoch {epoch})",
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
            "latent_dim": latent_dim,
            "embed_dim": embed_dim,
            "num_classes": num_classes,
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
    output_path: str = "samples/acgan_cifar10_samples.png",
    device_str: Optional[str] = None,
    latent_dim: int = 100,
    embed_dim: int = 100,
    num_classes: int = 10,
    use_ema: bool = False,
) -> None:
    """
    Loads saved AC-GAN generator weights and outputs canonical 10x10 class matrix.
    """
    device = get_device(device_str)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ema_path = checkpoint_path.replace("generator.pt", "generator_ema.pt")
    if use_ema and os.path.exists(ema_path):
        print(f"[Eval Grid] Loading EMA weights from: {ema_path}")
        checkpoint_path = ema_path

    state = torch.load(checkpoint_path, map_location=device)
    raw_weights = state["netG"] if "netG" in state else state

    netG = ACGANGenerator(latent_dim=latent_dim, embed_dim=embed_dim, num_classes=num_classes).to(device)
    netG.load_state_dict(raw_weights)
    netG.eval()

    fixed_classes = torch.arange(num_classes, device=device).repeat_interleave(10)
    fixed_z_base = torch.randn(10, latent_dim, device=device)
    fixed_z = fixed_z_base.repeat(num_classes, 1)

    with torch.no_grad():
        samples = netG(fixed_z, fixed_classes)

    save_image_grid(
        samples,
        output_path,
        nrow=10,
        value_range=(-1.0, 1.0),
        title="AC-GAN CIFAR-10 Evaluated Matrix",
    )
    print(f"[Generated Grid] Saved 10x10 class matrix to {output_path}")


# ==============================================================================
# 5. CLI ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="IF5152: Auxiliary Classifier GAN (AC-GAN) on CIFAR-10 (Odena et al. 2016)"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate (default: 0.0002)")
    parser.add_argument("--latent-dim", type=int, default=100, help="Latent noise vector dimension (default: 100)")
    parser.add_argument("--embed-dim", type=int, default=100, help="Class embedding dimension (default: 100)")
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
        chk = os.path.join(DEFAULT_CHECKPOINT_DIR, "acgan_cifar10_generator.pt")
        out = os.path.join(DEFAULT_SAMPLES_DIR, "acgan_cifar10_samples.png")
        generate_grid_from_checkpoint(
            chk,
            out,
            device_str=args.device,
            latent_dim=args.latent_dim,
            embed_dim=args.embed_dim,
            use_ema=args.use_ema,
        )
        return

    train_acgan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        embed_dim=args.embed_dim,
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
