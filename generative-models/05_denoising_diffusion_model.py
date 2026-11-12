"""
05_denoising_diffusion_model.py
===============================
Class-Conditional Denoising Diffusion Probabilistic Model (DDPM) & 
Denoising Diffusion Implicit Model (DDIM) on CIFAR-10 ($3 \\times 32 \\times 32$).

Key Concepts & Formulations (Ho et al. 2020; Song et al. 2021):
1. Forward Process (Noising):
   q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps, where eps ~ N(0, I)
2. Score Matching / Noise Prediction:
   Neural network epsilon_theta(x_t, t, y) predicts injected Gaussian noise.
3. Classifier-Free Guidance (CFG, Ho & Salimans, 2021):
   hat{eps}_theta(x_t, t, y) = eps_uncond + w * (eps_cond - eps_uncond)
   where null label emptyset is randomly dropped during training (p_uncond ~ 0.15).
4. DDIM Sampler (Song et al., ICLR 2021):
   Accelerated non-Markovian deterministic sampling (eta = 0) reducing sampling from
   T=1000 down to S=20-50 steps (~1-2 seconds per grid).
5. Architecture:
   Conditional ResNet U-Net with GroupNorm, SiLU, Multi-Head Self-Attention (16x16 and 8x8),
   and adaptive time/class conditioning injection.
"""

import argparse
import math
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
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
    "dog", "frog", "horse", "ship", "truck"
]


# ==============================================================================
# 1. Embeddings & U-Net Building Blocks
# ==============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal timestep embeddings (Vaswani et al. / Ho et al.).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AttentionBlock(nn.Module):
    """
    Multi-Head Spatial Self-Attention Block for 2D feature maps.
    Enables global spatial coordination across distant pixels.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_norm = self.norm(x)
        qkv = self.qkv(h_norm).view(b, 3, self.num_heads, self.head_dim, h * w)
        q = qkv[:, 0].transpose(-2, -1)  # (b, heads, hw, head_dim)
        k = qkv[:, 1].transpose(-2, -1)
        v = qkv[:, 2].transpose(-2, -1)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-2, -1).contiguous().view(b, c, h, w)
        return x + self.proj(out)


class ResBlock(nn.Module):
    """
    Residual Block with adaptive conditioning injection (time + class) and GroupNorm.
    """

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


# ==============================================================================
# 2. Conditional ResNet U-Net Architecture
# ==============================================================================

class ConditionalUNet(nn.Module):
    """
    Class-Conditional ResNet U-Net with Self-Attention for CIFAR-10 (3x32x32).
    Conditioning:
    - Sinusoidal timestep embedding + 2-layer MLP projection.
    - Learnable class embedding (num_classes + 1), where index num_classes is
      the null token for Classifier-Free Guidance.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_classes: int = 10,
        cond_dim: int = 128,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cond_dim = cond_dim

        # Timestep & Class embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        # Class embedding: [0, num_classes-1] are real classes; index num_classes is null token
        self.class_emb = nn.Embedding(num_classes + 1, cond_dim)

        # Initial convolution: 3x32x32 -> 64x32x32
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling hierarchy: 32 -> 16 -> 8
        self.down1_res = ResBlock(base_channels, base_channels, cond_dim)
        self.down1_conv = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1)  # 32 -> 16

        self.down2_res = ResBlock(base_channels, base_channels * 2, cond_dim)
        self.down2_attn = AttentionBlock(base_channels * 2)
        self.down2_conv = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1)  # 16 -> 8

        # Bottleneck: 8x8
        self.bot_res1 = ResBlock(base_channels * 2, base_channels * 2, cond_dim)
        self.bot_attn = AttentionBlock(base_channels * 2)
        self.bot_res2 = ResBlock(base_channels * 2, base_channels * 2, cond_dim)

        # Upsampling hierarchy: 8 -> 16 -> 32
        self.up1_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, bias=False),
        )  # 8 -> 16
        # Skip connection from down2 has base_channels * 2 -> total = 4 * base_channels
        self.up1_res = ResBlock(base_channels * 4, base_channels, cond_dim)
        self.up1_attn = AttentionBlock(base_channels)

        self.up2_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
        )  # 16 -> 32
        # Skip connection from down1 has base_channels -> total = 2 * base_channels
        self.up2_res = ResBlock(base_channels * 2, base_channels, cond_dim)

        # Final projection to RGB image
        self.norm_out = nn.GroupNorm(min(8, base_channels), base_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting injected Gaussian noise.
        - x: (B, C, H, W) noisy image batch
        - t: (B,) discrete timesteps in [0, T-1]
        - y: (B,) class labels in [0, num_classes] (num_classes is null token)
        """
        t_emb = self.time_mlp(t)
        y_emb = self.class_emb(y)
        cond = t_emb + y_emb

        x_in = self.conv_in(x)

        # Down 1
        d1 = self.down1_res(x_in, cond)
        p1 = self.down1_conv(d1)

        # Down 2
        d2 = self.down2_attn(self.down2_res(p1, cond))
        p2 = self.down2_conv(d2)

        # Bottleneck
        mid = self.bot_res1(p2, cond)
        mid = self.bot_attn(mid)
        mid = self.bot_res2(mid, cond)

        # Up 1
        u1 = self.up1_conv(mid)
        u1 = torch.cat([u1, d2], dim=1)
        h1 = self.up1_attn(self.up1_res(u1, cond))

        # Up 2
        u2 = self.up2_conv(h1)
        u2 = torch.cat([u2, d1], dim=1)
        h2 = self.up2_res(u2, cond)

        return self.conv_out(self.act_out(self.norm_out(h2)))


# ==============================================================================
# 3. Exponential Moving Average (EMA) Tracking
# ==============================================================================

class EMA:
    """
    Exponential Moving Average of model parameters (Song et al., Ho et al.).
    Maintains a shadow copy of the model weights:
        theta_ema <- decay * theta_ema + (1 - decay) * theta
    Crucial for diffusion models: smooths out high-frequency gradient variance,
    resulting in substantially higher perceptual quality, clearer textures, and lower FID.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: p.clone().detach()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> dict:
        """Loads shadow weights into model, returning original weights for subsequent training."""
        original = {}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                original[name] = p.clone().detach()
                p.data.copy_(self.shadow[name])
        return original

    def restore(self, model: nn.Module, original: dict) -> None:
        """Restores original weights into model after evaluation/sampling."""
        for name, p in model.named_parameters():
            if p.requires_grad and name in original:
                p.data.copy_(original[name])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict.get("decay", self.decay)
        loaded_shadow = state_dict.get("shadow", {})
        # Map device to current model device
        device = next(iter(self.shadow.values())).device if self.shadow else torch.device("cpu")
        self.shadow = {k: v.to(device) for k, v in loaded_shadow.items()}


# ==============================================================================
# 4. Diffusion Engine: Noise Schedules, DDPM & DDIM Samplers
# ==============================================================================

class GaussianDiffusion:
    """
    Engine managing forward noise process and dual reverse samplers:
    - DDPM: Stochastic ancestral reverse sampling (Ho et al., 2020)
    - DDIM: Fast deterministic ODE sampling (Song et al., ICLR 2021)
    - Classifier-Free Guidance (CFG, Ho & Salimans, 2021)
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        device: Optional[torch.device] = None,
    ):
        self.timesteps = timesteps
        self.device = device

        if beta_schedule == "linear":
            self.betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        elif beta_schedule == "cosine":
            # Improved cosine schedule (Nichol & Dhariwal, 2021)
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, device=device)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

        # Pre-computed coefficients for closed-form forward sampling
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat)

    def sample_forward(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Closed-form forward step: q(x_t | x_0) = sqrt(alpha_hat_t)*x_0 + sqrt(1 - alpha_hat_t)*eps
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_hat_t = self.sqrt_alpha_hat[t][:, None, None, None]
        sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t][:, None, None, None]
        x_t = sqrt_alpha_hat_t * x_0 + sqrt_one_minus_alpha_hat_t * noise
        return x_t, noise

    @torch.no_grad()
    def sample_ddim(
        self,
        model: nn.Module,
        labels: torch.Tensor,
        img_shape: Tuple[int, int, int] = (3, 32, 32),
        ddim_steps: int = 50,
        eta: float = 0.0,
        cfg_scale: float = 2.0,
    ) -> torch.Tensor:
        """
        Accelerated DDIM Sampling (Song et al., ICLR 2021) with Classifier-Free Guidance:
        x_{t-1} = sqrt(alpha_hat_{t-1}) * hat{x}_0 + sqrt(1 - alpha_hat_{t-1} - sigma_t^2) * eps_pred + sigma_t * z
        When eta = 0.0, the sampling trajectory is completely deterministic.
        """
        model.eval()
        b = labels.size(0)
        null_label = getattr(model, "num_classes", 10)
        x = torch.randn((b, *img_shape), device=self.device)

        # Select evenly spaced time steps in [0, T-1]
        c = self.timesteps // ddim_steps
        time_steps = torch.arange(0, self.timesteps, c, device=self.device).long()
        # Append latest step if not present
        if time_steps[-1] != self.timesteps - 1:
            time_steps = torch.cat([time_steps, torch.tensor([self.timesteps - 1], device=self.device)])
        time_steps_prev = torch.cat([torch.tensor([-1], device=self.device), time_steps[:-1]])

        null_labels = torch.full_like(labels, null_label)

        for i in reversed(range(len(time_steps))):
            t_val = time_steps[i]
            t_prev_val = time_steps_prev[i]
            t = torch.full((b,), t_val, device=self.device, dtype=torch.long)

            # Model prediction with Classifier-Free Guidance
            if cfg_scale > 1.0:
                # Batched joint evaluation
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([t, t], dim=0)
                y_in = torch.cat([labels, null_labels], dim=0)
                eps_both = model(x_in, t_in, y_in)
                eps_cond, eps_uncond = eps_both.chunk(2, dim=0)
                eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = model(x, t, labels)

            alpha_hat_t = self.alpha_hat[t_val]
            alpha_hat_prev = self.alpha_hat[t_prev_val] if t_prev_val >= 0 else torch.tensor(1.0, device=self.device)

            # 1. Predict clean image hat{x}_0
            pred_x0 = (x - torch.sqrt(1.0 - alpha_hat_t) * eps_pred) / torch.sqrt(alpha_hat_t)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # 2. Compute variance sigma_t
            sigma_t = eta * torch.sqrt((1.0 - alpha_hat_prev) / (1.0 - alpha_hat_t) * (1.0 - alpha_hat_t / alpha_hat_prev))

            # 3. Direction pointing to x_t
            pred_dir = torch.sqrt(torch.clamp(1.0 - alpha_hat_prev - sigma_t ** 2, min=0.0)) * eps_pred

            # 4. Step backward
            noise = torch.randn_like(x) if sigma_t > 0 else 0.0
            x = torch.sqrt(alpha_hat_prev) * pred_x0 + pred_dir + sigma_t * noise

        return x

    @torch.no_grad()
    def sample_ddpm(
        self,
        model: nn.Module,
        labels: torch.Tensor,
        img_shape: Tuple[int, int, int] = (3, 32, 32),
        cfg_scale: float = 2.0,
    ) -> torch.Tensor:
        """
        Full stochastic ancestral DDPM sampling (T steps).
        """
        model.eval()
        b = labels.size(0)
        null_label = getattr(model, "num_classes", 10)
        x = torch.randn((b, *img_shape), device=self.device)
        null_labels = torch.full_like(labels, null_label)

        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)

            if cfg_scale > 1.0:
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([t, t], dim=0)
                y_in = torch.cat([labels, null_labels], dim=0)
                eps_both = model(x_in, t_in, y_in)
                eps_cond, eps_uncond = eps_both.chunk(2, dim=0)
                eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = model(x, t, labels)

            alpha = self.alphas[i]
            alpha_hat = self.alpha_hat[i]
            beta = self.betas[i]

            noise = torch.randn_like(x) if i > 0 else 0.0
            x = (1.0 / torch.sqrt(alpha)) * (
                x - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_hat)) * eps_pred
            ) + torch.sqrt(beta) * noise

        return x


# ==============================================================================
# 4. Training Pipeline & Evaluation
# ==============================================================================

def train_diffusion(args):
    device = get_device(args.device)
    print("=" * 65)
    print("IF5152: Class-Conditional Diffusion Model (CIFAR-10)")
    print(f"Device        : {device}")
    print(f"Timesteps (T) : {args.timesteps} ({args.beta_schedule} schedule)")
    print(f"Sampler       : {args.sampler.upper()} (DDIM Steps: {args.ddim_steps}, CFG: {args.cfg_scale})")
    print(f"Batch Size    : {args.batch_size} | Epochs : {args.epochs}")
    print(f"Learning Rate : {args.lr} | CFG Drop p : {args.p_uncond}")
    print("=" * 65)

    # 1. DataLoader
    train_loader = get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train=True,
        normalize_range=(-1.0, 1.0),
        quick_test=args.quick_test,
        max_samples=1000,
    )
    print(f"Training batches per epoch: {len(train_loader)}")

    # 2. Instantiate Model & Diffusion Engine
    num_classes = len(CIFAR10_CLASSES) if args.dataset.lower() in ["cifar", "cifar10", "cifar-10"] else 10
    model = ConditionalUNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        num_classes=num_classes,
        cond_dim=args.cond_dim,
    ).to(device)

    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        device=device,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    ema = EMA(model, decay=args.ema_decay)

    # Paths
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    full_ckpt_path = os.path.join(args.checkpoint_dir, f"cddpm_{args.dataset}_checkpoint.pt")
    model_ckpt_path = os.path.join(args.checkpoint_dir, f"cddpm_{args.dataset}_model.pt")

    start_epoch = 1

    # Resume
    if args.resume and os.path.exists(full_ckpt_path):
        print(f"[Resume] Loading checkpoint from {full_ckpt_path}...")
        ckpt = torch.load(full_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
            print("[Resume] Restored EMA shadow parameters.")
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] Resuming at epoch {start_epoch}.")

    # Fixed evaluation probe (10 classes x 10 samples = 100 images)
    torch.manual_seed(42)
    fixed_labels = torch.arange(num_classes, device=device).repeat_interleave(10)

    # Sample-only mode
    if args.sample_only:
        if os.path.exists(model_ckpt_path):
            model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
        elif os.path.exists(full_ckpt_path):
            ckpt = torch.load(full_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            print(f"[Error] No checkpoint found in {args.checkpoint_dir}. Train the model first.")
            sys.exit(1)

        print(f"[Sample Only] Generating 10x10 evaluation grid using {args.sampler.upper()}...")
        if args.sampler.lower() == "ddim":
            samples = diffusion.sample_ddim(
                model=model,
                labels=fixed_labels,
                img_shape=(3, 32, 32),
                ddim_steps=args.ddim_steps,
                eta=args.ddim_eta,
                cfg_scale=args.cfg_scale,
            )
        else:
            samples = diffusion.sample_ddpm(
                model=model,
                labels=fixed_labels,
                img_shape=(3, 32, 32),
                cfg_scale=args.cfg_scale,
            )

        out_path = os.path.join(args.output_dir, f"cddpm_{args.dataset}_samples.png")
        save_image_grid(
            samples,
            filename=out_path,
            nrow=10,
            value_range=(-1.0, 1.0),
            title=f"Conditional Diffusion ({args.dataset.upper()} 32x32) DDIM Samples (CFG={args.cfg_scale})",
        )
        print(f"[Sample Only] Generated grid saved to: {out_path}")
        return

    # 3. Training Loop
    print(f"\nStarting training for {args.epochs} epoch(s)...")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            b = images.size(0)

            optimizer.zero_grad()

            # Randomly sample discrete timesteps t in [0, T-1]
            t = torch.randint(0, args.timesteps, (b,), device=device).long()

            # Classifier-Free Guidance training: randomly drop labels to null token with prob p_uncond
            drop_mask = torch.rand(b, device=device) < args.p_uncond
            labels_train = labels.clone()
            labels_train[drop_mask] = num_classes  # null token index

            # Forward noise injection
            x_t, noise = diffusion.sample_forward(images, t)

            # Predict noise
            predicted_noise = model(x_t, t, labels_train)

            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            ema.update(model)

            total_loss += loss.item()
            pbar.set_postfix({"Diffusion MSE": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.epochs} Complete | Average MSE: {avg_loss:.4f}")

        # Milestone visual generation via DDIM using smoothed EMA parameters
        if args.save_milestones and (epoch in [1, 5, 10, 15, 20, 25, 30, 40, 50] or epoch == args.epochs):
            print(f"[Milestone] Generating 10x10 sample grid via DDIM (EMA shadow weights, S={args.ddim_steps})...")
            orig_weights = ema.apply_shadow(model)
            sample_grid = diffusion.sample_ddim(
                model=model,
                labels=fixed_labels,
                img_shape=(3, 32, 32),
                ddim_steps=args.ddim_steps,
                eta=args.ddim_eta,
                cfg_scale=args.cfg_scale,
            )
            ema.restore(model, orig_weights)

            out_path = os.path.join(args.output_dir, f"cddpm_{args.dataset}_samples.png")
            save_image_grid(
                sample_grid,
                filename=out_path,
                nrow=10,
                value_range=(-1.0, 1.0),
                title=f"Conditional Diffusion ({args.dataset.upper()} 32x32) Epoch {epoch}/{args.epochs} (CFG={args.cfg_scale})",
            )
            milestone_dir = os.path.join(args.output_dir, f"cddpm_{args.dataset}_milestones")
            os.makedirs(milestone_dir, exist_ok=True)
            milestone_path = os.path.join(milestone_dir, f"epoch_{epoch:02d}.png")
            save_image_grid(
                sample_grid,
                filename=milestone_path,
                nrow=10,
                value_range=(-1.0, 1.0),
                title=f"Conditional Diffusion Epoch {epoch}/{args.epochs} (EMA)",
            )

        # Checkpointing: Save smoothed EMA weights into model_ckpt_path for high quality downstream inference
        orig_weights = ema.apply_shadow(model)
        torch.save(model.state_dict(), model_ckpt_path)
        ema.restore(model, orig_weights)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }, full_ckpt_path)

    print(f"\n[Done] Training complete for {args.dataset}!")
    print(f"Samples saved to     : {os.path.join(args.output_dir, f'cddpm_{args.dataset}_samples.png')}")
    print(f"Model saved to       : {model_ckpt_path}")
    print(f"Checkpoint saved to  : {full_ckpt_path}")


# ==============================================================================
# 5. CLI Entry Point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IF5152: Class-Conditional Diffusion Model with DDIM Sampler")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar", "mnist"])
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--timesteps", type=int, default=1000, help="Total forward diffusion steps (T)")
    parser.add_argument("--beta-schedule", type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--base-channels", type=int, default=48, help="Base channel multiplier for ResNet U-Net")
    parser.add_argument("--cond-dim", type=int, default=128, help="Embedding dimension for time and class")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for AdamW")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="Decay factor for Exponential Moving Average (EMA)")
    parser.add_argument("--p-uncond", type=float, default=0.15, help="Probability of dropping class label for CFG")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"], help="Sampling algorithm")
    parser.add_argument("--ddim-steps", type=int, default=50, help="Number of reverse steps for DDIM sampler")
    parser.add_argument("--ddim-eta", type=float, default=0.0, help="DDIM stochasticity eta (0.0 = deterministic ODE)")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="Classifier-Free Guidance weight (w >= 1.0)")
    parser.add_argument("--device", type=str, default=None, help="Compute device (mps/cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_SAMPLES_DIR, help="Directory to save sample grids")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="Directory to save checkpoints")
    parser.add_argument("--save-milestones", action="store_true", default=True, help="Save milestone snapshots")
    parser.add_argument("--quick-test", action="store_true", help="Quick run on small subset for verification")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint")
    parser.add_argument("--sample-only", action="store_true", help="Generate sample grid from saved checkpoint")
    args = parser.parse_args()

    train_diffusion(args)
