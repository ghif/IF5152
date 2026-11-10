"""
06_rectified_flow_cifar10.py
============================
Class-Conditional Rectified Flow on CIFAR-10 ($3 \times 32 \times 32$, 10 Classes).

Theoretical Foundations (IF5152 Lecture Slides, Pages 180-181 & 204-212):
1. Generalized Diffusion: Rectified Flow (Liu et al. ICLR 2023; Lipman et al. 2023):
   - Rather than simulating curved SDE diffusion paths (DDPM/VP/VE), Rectified Flow
     connects data x ~ p_data and Gaussian noise z ~ N(0, I) via a straight linear trajectory:
         x_t = (1 - t) * x + t * z,   where t in [0, 1]
     - At t = 0: x_0 = x  (authentic CIFAR-10 data)
     - At t = 1: x_1 = z  (standard normal prior noise)

2. Ground-Truth Velocity Field:
   - Differentiating the linear path with respect to time yields a constant velocity:
         v_gt = d(x_t)/dt = z - x

3. Neural Velocity Matching Objective:
   - A parameterized neural network v_theta(x_t, t, y) learns the vector field:
         L_RF(theta) = E_{x, z, t, y} [ || v_theta(x_t, t, y) - (z - x) ||_2^2 ]

4. Classifier-Free Guidance (CFG) in Velocity Space (Pages 161-169):
   - During training: Random label dropout (p_uncond = 0.15) substituting null token y_null = 10.
   - During inference: Extrapolate velocity along the conditioned direction:
         v_cfg = v_theta(x_t, t, y_null) + s * (v_theta(x_t, t, y) - v_theta(x_t, t, y_null))
     where s = 1 + w is the guidance scale (e.g. s = 2.0).

5. Reverse ODE Integration (Pages 204-208):
   - Sampling starts from pure noise x_1 ~ N(0, I) at t = 1 and integrates backwards to t = 0:
         dx_t / dt = v_theta(x_t, t, y)  =>  x_{t - dt} = x_t - dt * v_cfg
   - Supported Numerical Solvers:
     - 1st-Order Euler Solver
     - 2nd-Order Heun (Midpoint Predictor-Corrector) Solver
"""

import argparse
import copy
import math
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


# ==============================================================================
# 1. EMBEDDINGS & NEURAL NETWORK MODULES
# ==============================================================================
class SinusoidalTimeEmbedding(nn.Module):
    """
    Continuous sinusoidal time embeddings for t in [0, 1].
    Maps scalar time to a high-dimensional harmonic representation.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Scale t from [0, 1] to [0, 1000] to match standard diffusion frequencies
        t_scaled = t * 1000.0
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=device)
            / half_dim
        )
        args = t_scaled[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class AttentionBlock(nn.Module):
    """
    Multi-Head Spatial Self-Attention for 2D feature maps.
    Enables long-range spatial coordination across image pixels.
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

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-2, -1).contiguous().view(b, c, h, w)
        return x + self.proj(out)


class ConditionalResBlock(nn.Module):
    """
    Residual Block with Adaptive Conditioning for Time and Class.
    Injects conditioning features via linear projection and affine modulation.
    """

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # Inject conditioning as spatial broadcast
        cond = self.cond_proj(cond_emb)[:, :, None, None]
        h = h + cond
        h = self.conv2(F.silu(self.norm2(h)))
        return self.shortcut(x) + h


class RectifiedFlowUNet(nn.Module):
    """
    U-Net Velocity Field Predictor for CIFAR-10 (3x32x32).
    Predicts v_theta(x_t, t, y) in R^{3 x 32 x 32}.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_classes: int = 10,
        cond_dim: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_classes = num_classes
        self.null_class = num_classes  # Index 10 is unconditional null token

        # Time and Class embeddings
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.class_embed = nn.Embedding(num_classes + 1, cond_dim)

        c = base_channels
        # Initial convolution: 3x32x32 -> c x 32x32
        self.conv_in = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)

        # Encoder stages:
        # Stage 1: 32x32
        self.enc1 = ConditionalResBlock(c, c, cond_dim)
        self.down1 = nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)  # 32x32 -> 16x16

        # Stage 2: 16x16 (with Self-Attention)
        self.enc2 = ConditionalResBlock(c * 2, c * 2, cond_dim)
        self.attn2 = AttentionBlock(c * 2)
        self.down2 = nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)  # 16x16 -> 8x8

        # Stage 3: 8x8 (with Self-Attention)
        self.enc3 = ConditionalResBlock(c * 4, c * 4, cond_dim)
        self.attn3 = AttentionBlock(c * 4)
        self.down3 = nn.Conv2d(c * 4, c * 4, kernel_size=4, stride=2, padding=1)  # 8x8 -> 4x4

        # Bottleneck: 4x4
        self.mid_res1 = ConditionalResBlock(c * 4, c * 4, cond_dim)
        self.mid_attn = AttentionBlock(c * 4)
        self.mid_res2 = ConditionalResBlock(c * 4, c * 4, cond_dim)

        # Decoder stages:
        # Stage 3 Up: 4x4 -> 8x8
        self.up3 = nn.ConvTranspose2d(c * 4, c * 4, kernel_size=4, stride=2, padding=1)
        self.dec3 = ConditionalResBlock(c * 8, c * 4, cond_dim)
        self.attn_dec3 = AttentionBlock(c * 4)

        # Stage 2 Up: 8x8 -> 16x16
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConditionalResBlock(c * 4, c * 2, cond_dim)
        self.attn_dec2 = AttentionBlock(c * 2)

        # Stage 1 Up: 16x16 -> 32x32
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConditionalResBlock(c * 2, c, cond_dim)

        # Output projection
        self.norm_out = nn.GroupNorm(min(8, c), c)
        self.conv_out = nn.Conv2d(c, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predicts velocity field v_theta(x, t, y).
        t: (batch,) in range [0, 1]
        y: (batch,) class indices in range [0, 10]
        """
        # 1. Conditioning embedding
        t_emb = self.time_embed(t)
        if y is None:
            y = torch.full((x.size(0),), self.null_class, device=x.device, dtype=torch.long)
        c_emb = self.class_embed(y)
        cond = t_emb + c_emb

        # 2. Encoder
        h0 = self.conv_in(x)
        h1 = self.enc1(h0, cond)
        d1 = self.down1(h1)

        h2 = self.attn2(self.enc2(d1, cond))
        d2 = self.down2(h2)

        h3 = self.attn3(self.enc3(d2, cond))
        d3 = self.down3(h3)

        # 3. Bottleneck
        m = self.mid_res1(d3, cond)
        m = self.mid_attn(m)
        m = self.mid_res2(m, cond)

        # 4. Decoder with skip connections
        u3 = self.up3(m)
        u3 = torch.cat([u3, h3], dim=1)
        u3 = self.attn_dec3(self.dec3(u3, cond))

        u2 = self.up2(u3)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.attn_dec2(self.dec2(u2, cond))

        u1 = self.up1(u2)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.dec1(u1, cond)

        out = self.conv_out(F.silu(self.norm_out(u1)))
        return out


# ==============================================================================
# 2. RECTIFIED FLOW ENGINE (TRAINING & ODE SAMPLING)
# ==============================================================================
class RectifiedFlow:
    """
    Rectified Flow Engine (Liu et al. 2023; Esser et al. 2024).
    Straight linear interpolation: x_t = (1 - t) * x + t * z
    Target velocity field: v_gt = z - x
    """

    def __init__(self, time_schedule: str = "uniform", num_classes: int = 10):
        self.time_schedule = time_schedule
        self.num_classes = num_classes
        self.null_class = num_classes

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Samples continuous timesteps t in [0, 1].
        Supports uniform or logit-normal distribution (Slide 180).
        """
        if self.time_schedule == "logit_normal":
            # Logit-normal puts more emphasis on ambiguous middle noise
            u = torch.randn(batch_size, device=device)
            t = torch.sigmoid(u)
        else:
            # Standard uniform distribution: t ~ U(0, 1)
            t = torch.rand(batch_size, device=device)
        return t

    def get_train_tuple(
        self, x: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Constructs training tuple:
            x_t = (1 - t) * x + t * z
            v_gt = z - x
        """
        b = x.size(0)
        z = torch.randn_like(x)
        t = self.sample_time(b, device)
        t_spatial = t[:, None, None, None]

        # Straight-line linear interpolation path
        x_t = (1.0 - t_spatial) * x + t_spatial * z
        v_gt = z - x

        return x_t, t, v_gt, z

    def compute_loss(
        self,
        net: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        p_uncond: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes velocity matching MSE loss with Classifier-Free Guidance dropout.
        """
        device = x.device
        b = x.size(0)
        x_t, t, v_gt, _ = self.get_train_tuple(x, device)

        # Apply random label dropout for Classifier-Free Guidance (Slides 161-164)
        if p_uncond > 0:
            mask = torch.rand(b, device=device) < p_uncond
            y_train = y.clone()
            y_train[mask] = self.null_class
        else:
            y_train = y

        v_pred = net(x_t, t, y_train)
        loss = F.mse_loss(v_pred, v_gt)
        return loss, v_pred

    @torch.no_grad()
    def sample_ode(
        self,
        net: nn.Module,
        shape: Tuple[int, int, int, int],
        classes: torch.Tensor,
        steps: int = 50,
        cfg_scale: float = 2.0,
        solver: str = "euler",
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generates synthetic images by numerically integrating the reverse ODE
        from pure noise x_1 ~ N(0, I) at t=1 down to clean data x_0 at t=0:
            dx_t / dt = v_theta(x_t, t, y)
        """
        device = device or classes.device
        b = shape[0]
        # Start at t = 1 from pure Gaussian noise
        x = torch.randn(shape, device=device)

        # Discrete timesteps from 1.0 down to 0.0
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

        null_classes = torch.full((b,), self.null_class, device=device, dtype=torch.long)

        for i in range(steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_curr - t_next  # Positive step size (dt > 0)
            t_batch = torch.full((b,), t_curr.item(), device=device)

            # Evaluate conditioned and unconditioned velocity
            if cfg_scale > 1.0:
                v_cond = net(x, t_batch, classes)
                v_uncond = net(x, t_batch, null_classes)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = net(x, t_batch, classes)

            if solver == "heun" and i < steps - 1:
                # 2nd-Order Heun (Predictor-Corrector) ODE step
                x_pred = x - dt * v
                t_next_batch = torch.full((b,), t_next.item(), device=device)
                if cfg_scale > 1.0:
                    v_next_cond = net(x_pred, t_next_batch, classes)
                    v_next_uncond = net(x_pred, t_next_batch, null_classes)
                    v_next = v_next_uncond + cfg_scale * (v_next_cond - v_next_uncond)
                else:
                    v_next = net(x_pred, t_next_batch, classes)
                x = x - 0.5 * dt * (v + v_next)
            else:
                # 1st-Order Euler ODE step: x_{t - dt} = x_t - dt * v
                x = x - dt * v

        # Clamping output to valid normalized image range [-1, 1]
        return x.clamp(-1.0, 1.0)


# ==============================================================================
# 3. EXPONENTIAL MOVING AVERAGE (EMA) TRACKER
# ==============================================================================
class EMAModel:
    """
    Exponential Moving Average (EMA) tracker for model parameters.
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
def train_rectified_flow(
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.0002,
    base_channels: int = 64,
    time_schedule: str = "uniform",
    p_uncond: float = 0.15,
    sample_steps: int = 25,
    cfg_scale: float = 2.0,
    solver: str = "euler",
    use_ema: bool = True,
    ema_decay: float = 0.999,
    use_fp16: bool = True,
    device_str: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    samples_dir: Optional[str] = None,
    quick_test: bool = False,
    resume: bool = False,
) -> Tuple[RectifiedFlowUNet, RectifiedFlow]:
    """
    Executes training loop for Class-Conditional Rectified Flow on CIFAR-10.
    Supports CUDA FP16 Mixed Precision on NVIDIA T4/G4/A100 GPUs and MPS on Apple Silicon.
    """
    device = get_device(device_str)
    print(f"[Device] Using {device} for Rectified Flow CIFAR-10 training.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[CUDA Accelerator] Active GPU: {gpu_name} (T4/G4 Tensor Core acceleration)")

    checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    samples_dir = samples_dir or DEFAULT_SAMPLES_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    milestones_dir = os.path.join(samples_dir, "rectified_flow_cifar10_milestones")
    os.makedirs(milestones_dir, exist_ok=True)

    chk_path = os.path.join(checkpoint_dir, "rectified_flow_cifar10_checkpoint.pt")
    model_path = os.path.join(checkpoint_dir, "rectified_flow_cifar10_model.pt")
    model_ema_path = os.path.join(checkpoint_dir, "rectified_flow_cifar10_model_ema.pt")

    num_classes = 10

    # 1. Instantiate network & engine
    net = RectifiedFlowUNet(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        num_classes=num_classes,
    ).to(device)

    rf = RectifiedFlow(time_schedule=time_schedule, num_classes=num_classes)
    ema_net = EMAModel(net, decay=ema_decay) if use_ema else None

    # 2. Optimizer, Mixed Precision Scaler & Cosine Annealing Scheduler
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    use_amp = (device.type == "cuda") and use_fp16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    start_epoch = 1

    if resume and os.path.exists(chk_path):
        print(f"[Resume] Loading Rectified Flow checkpoint from {chk_path}...")
        chk = torch.load(chk_path, map_location=device)
        net.load_state_dict(chk["net"])
        optimizer.load_state_dict(chk["optimizer"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr
            pg["initial_lr"] = lr
        if ema_net is not None and "ema_net" in chk:
            ema_net.load_state_dict(chk["ema_net"])
        start_epoch = chk["epoch"] + 1
        print(f"[Resume] Resuming at Epoch {start_epoch}/{epochs} (Previous Loss={chk.get('loss', 0):.4f})")

    total_remaining = max(1, epochs - start_epoch + 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_remaining, eta_min=1e-6)

    # 3. Fixed probe: 10 classes (rows) x 10 distinct noise vectors (columns) = 100 images
    fixed_classes = torch.arange(num_classes, device=device).repeat_interleave(10)

    if quick_test:
        epochs = min(epochs, start_epoch)
        batch_size = min(batch_size, 16)
        sample_steps = 5

    # 4. Data loader
    dataloader = get_dataloader(
        dataset_name="cifar10",
        batch_size=batch_size,
        train=True,
        normalize_range=(-1.0, 1.0),
        quick_test=quick_test,
        max_samples=50 if quick_test else 50000,
    )

    print(f"\n========================================================")
    print(f"  Class-Conditional Rectified Flow on CIFAR-10          ")
    print(f"  Reference: IF5152 Slides (Pages 180-181 & 204-212)    ")
    print(f"========================================================")
    print(f"  Epochs           : {epochs} (starting at {start_epoch})")
    print(f"  Batch Size       : {batch_size}")
    print(f"  Base Channels    : {base_channels}")
    print(f"  Time Schedule    : {time_schedule}")
    print(f"  CFG Dropout      : p_uncond = {p_uncond}")
    print(f"  Mixed Precision  : {'FP16 (Tensor Cores)' if use_amp else 'FP32'}")
    print(f"  Sampling Solver  : {solver.upper()} ({sample_steps} steps, CFG={cfg_scale})")
    print(f"  EMA Tracking     : {'Active (decay=' + str(ema_decay) + ')' if use_ema else 'Disabled'}")
    print(f"  Checkpoints Dir  : {checkpoint_dir}")
    print(f"  Samples Dir      : {samples_dir}")
    print(f"  Total Batches    : {len(dataloader)}")
    print(f"========================================================\n")

    for epoch in range(start_epoch, epochs + 1):
        net.train()
        running_loss = 0.0
        batch_count = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{epochs:02d}", leave=True)

        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    loss, _ = rf.compute_loss(net, imgs, labels, p_uncond=p_uncond)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, _ = rf.compute_loss(net, imgs, labels, p_uncond=p_uncond)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            if ema_net is not None:
                ema_net.update(net)

            running_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{optimizer.param_groups[0]['lr']:.2e}"})

        scheduler.step()
        avg_loss = running_loss / max(1, batch_count)
        print(f"[Epoch {epoch:02d}] Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Evaluation & Milestone Snapshot
        net.eval()
        active_model = net
        if ema_net is not None:
            with ema_net.apply_shadow(net):
                samples = rf.sample_ode(
                    net,
                    shape=(100, 3, 32, 32),
                    classes=fixed_classes,
                    steps=sample_steps,
                    cfg_scale=cfg_scale,
                    solver=solver,
                    device=device,
                )
        else:
            samples = rf.sample_ode(
                net,
                shape=(100, 3, 32, 32),
                classes=fixed_classes,
                steps=sample_steps,
                cfg_scale=cfg_scale,
                solver=solver,
                device=device,
            )

        samples_path = os.path.join(samples_dir, "rectified_flow_cifar10_samples.png")
        save_image_grid(
            samples,
            samples_path,
            nrow=10,
            value_range=(-1.0, 1.0),
            title=f"Rectified Flow CIFAR-10 Grid (Epoch {epoch:02d})",
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            m_path = os.path.join(milestones_dir, f"epoch_{epoch:02d}.png")
            save_image_grid(
                samples,
                m_path,
                nrow=10,
                value_range=(-1.0, 1.0),
                title=f"Rectified Flow Milestone (Epoch {epoch:02d})",
            )
            print(f"[Milestone] Saved snapshot: {m_path}")

        # Save Checkpoints
        chk_dict = {
            "epoch": epoch,
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg_loss,
            "base_channels": base_channels,
            "time_schedule": time_schedule,
            "num_classes": num_classes,
        }
        if ema_net is not None:
            chk_dict["ema_net"] = ema_net.state_dict()
            with ema_net.apply_shadow(net):
                torch.save(net.state_dict(), model_ema_path)

        torch.save(chk_dict, chk_path)
        torch.save(net.state_dict(), model_path)

    print(f"\n[Training Complete] Final weights saved to:")
    print(f"  Model (Raw): {model_path}")
    if ema_net is not None:
        print(f"  Model (EMA): {model_ema_path}")
    print(f"  Full State : {chk_path}")
    print(f"  Samples    : {samples_path}\n")

    return net, rf


def generate_grid_from_checkpoint(
    checkpoint_path: str,
    output_path: str = "samples/rectified_flow_cifar10_samples.png",
    device_str: Optional[str] = None,
    base_channels: int = 64,
    steps: int = 25,
    cfg_scale: float = 2.0,
    solver: str = "euler",
    use_ema: bool = False,
) -> None:
    """
    Loads saved Rectified Flow weights and generates canonical 10x10 class matrix via ODE integration.
    """
    device = get_device(device_str)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ema_path = checkpoint_path.replace("model.pt", "model_ema.pt")
    if use_ema and os.path.exists(ema_path):
        print(f"[Eval Grid] Loading EMA weights from: {ema_path}")
        checkpoint_path = ema_path

    state = torch.load(checkpoint_path, map_location=device)
    raw_weights = state["net"] if isinstance(state, dict) and "net" in state else state

    num_classes = 10
    net = RectifiedFlowUNet(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        num_classes=num_classes,
    ).to(device)
    net.load_state_dict(raw_weights, strict=False)
    net.eval()

    rf = RectifiedFlow(num_classes=num_classes)
    fixed_classes = torch.arange(num_classes, device=device).repeat_interleave(10)

    print(f"[Eval Grid] Generating 10x10 matrix using {solver.upper()} solver ({steps} steps, CFG={cfg_scale})...")
    samples = rf.sample_ode(
        net,
        shape=(100, 3, 32, 32),
        classes=fixed_classes,
        steps=steps,
        cfg_scale=cfg_scale,
        solver=solver,
        device=device,
    )

    save_image_grid(
        samples,
        output_path,
        nrow=10,
        value_range=(-1.0, 1.0),
        title=f"Rectified Flow CIFAR-10 Grid ({solver.upper()} - {steps} Steps)",
    )
    print(f"[Generated Grid] Saved 10x10 class matrix to {output_path}")


# ==============================================================================
# 5. CLI ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="IF5152: Class-Conditional Rectified Flow on CIFAR-10 (Lecture Slides 180-181 & 204-212)"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate (default: 0.0002)")
    parser.add_argument("--channels", type=int, default=64, help="Base U-Net channels (default: 64)")
    parser.add_argument("--time-schedule", type=str, default="uniform", choices=["uniform", "logit_normal"], help="Time sampling distribution (default: uniform)")
    parser.add_argument("--p-uncond", type=float, default=0.15, help="CFG label dropout probability (default: 0.15)")
    parser.add_argument("--steps", type=int, default=25, help="ODE sampling steps (default: 25)")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="Classifier-Free Guidance scale (default: 2.0)")
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "heun"], help="ODE integration solver (default: euler)")
    parser.add_argument("--no-ema", action="store_true", help="Disable Exponential Moving Average tracking")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay rate (default: 0.999)")
    parser.add_argument("--no-fp16", action="store_true", help="Disable CUDA mixed precision FP16 (enabled by default on NVIDIA GPUs)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save checkpoints (default: checkpoints/)")
    parser.add_argument("--samples-dir", type=str, default=None, help="Directory to save generated sample images (default: samples/)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--quick-test", action="store_true", help="Quick smoke test (1 epoch, 50 samples)")
    parser.add_argument("--eval-grid", action="store_true", help="Generate 10x10 class grid from saved checkpoint")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights for evaluation grid")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    samples_dir = args.samples_dir or DEFAULT_SAMPLES_DIR

    if args.eval_grid:
        chk = os.path.join(checkpoint_dir, "rectified_flow_cifar10_model.pt")
        out = os.path.join(samples_dir, "rectified_flow_cifar10_samples.png")
        generate_grid_from_checkpoint(
            chk,
            out,
            device_str=args.device,
            base_channels=args.channels,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            solver=args.solver,
            use_ema=args.use_ema,
        )
        return

    train_rectified_flow(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_channels=args.channels,
        time_schedule=args.time_schedule,
        p_uncond=args.p_uncond,
        sample_steps=args.steps,
        cfg_scale=args.cfg_scale,
        solver=args.solver,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        use_fp16=not args.no_fp16,
        device_str=args.device,
        checkpoint_dir=args.checkpoint_dir,
        samples_dir=args.samples_dir,
        quick_test=args.quick_test,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
