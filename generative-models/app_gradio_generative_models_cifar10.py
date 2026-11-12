"""
app_gradio_generative_models_cifar10.py
======================================
Interactive Gradio Interface for IF5152: Generative Models Laboratory.
Unifying 4 Major CIFAR-10 Generative Paradigms:
1. 🎲 DCGAN: Deep Convolutional GAN (Radford et al. 2016)
2. 🎯 cGAN: Class-Conditional DCGAN (Mirza & Osindero 2014)
3. 🏗️ ResNet AC-GAN: Auxiliary Classifier GAN with Spectral Normalization & EMA (Odena et al. 2017)
4. 🌊 Continuous Generative Models:
   - Rectified Flow / Flow Matching (Liu et al. 2022 / InstaFlow)
   - Conditional Denoising Diffusion (DDPM / DDIM, Ho et al. 2020 / Song et al. 2021)
5. 📈 Evolution & Milestones: Multi-epoch progression gallery (Epochs 1-150) and comparative taxonomy.
"""

import argparse
import importlib.util
import math
import os
import random
import sys
import time
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

# Ensure generative-models directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils import DEFAULT_DATA_DIR, DEFAULT_SAMPLES_DIR, get_device

# Checkpoint paths
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
DCGAN_GEN_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "dcgan_cifar10_generator.pt")
DCGAN_DISC_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "dcgan_cifar10_discriminator.pt")
CGAN_GEN_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "cgan_cifar10_generator.pt")
CGAN_DISC_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "cgan_cifar10_discriminator.pt")
RESNET_ACGAN_GEN_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "resnet_acgan_cifar10_generator.pt")
RESNET_ACGAN_GEN_EMA_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "resnet_acgan_cifar10_generator_ema.pt")
RESNET_ACGAN_DISC_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "resnet_acgan_cifar10_discriminator.pt")
RECTIFIED_FLOW_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "rectified_flow_cifar10_model.pt")
RECTIFIED_FLOW_MODEL_EMA_PATH = os.path.join(CHECKPOINT_DIR, "rectified_flow_cifar10_model_ema.pt")
CDDPM_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "cddpm_cifar10_model.pt")

DCGAN_MILESTONES_DIR = os.path.join(DEFAULT_SAMPLES_DIR, "cifar10_milestones")
CGAN_MILESTONES_DIR = os.path.join(DEFAULT_SAMPLES_DIR, "cgan_cifar10_milestones")
RESNET_ACGAN_MILESTONES_DIR = os.path.join(DEFAULT_SAMPLES_DIR, "resnet_acgan_cifar10_milestones")
RECTIFIED_FLOW_MILESTONES_DIR = os.path.join(DEFAULT_SAMPLES_DIR, "rectified_flow_cifar10_milestones")
CDDPM_MILESTONES_DIR = os.path.join(DEFAULT_SAMPLES_DIR, "cddpm_cifar10_milestones")

RECTIFIED_FLOW_SAMPLE_GRID_PATH = os.path.join(DEFAULT_SAMPLES_DIR, "rectified_flow_cifar10_samples_ema_heun25.png")
RECTIFIED_FLOW_SAMPLE_RAW_PATH = os.path.join(DEFAULT_SAMPLES_DIR, "rectified_flow_cifar10_samples.png")
CDDPM_SAMPLE_GRID_PATH = os.path.join(DEFAULT_SAMPLES_DIR, "cddpm_cifar10_samples.png")
RESNET_ACGAN_SAMPLE_GRID_PATH = os.path.join(DEFAULT_SAMPLES_DIR, "resnet_acgan_cifar10_samples_ema.png")
CGAN_SAMPLE_GRID_PATH = os.path.join(DEFAULT_SAMPLES_DIR, "cgan_cifar10_samples.png")
DCGAN_SAMPLE_GRID_PATH = os.path.join(DEFAULT_SAMPLES_DIR, "gan_samples_cifar10.png")

LATENT_DIM = 100
RESNET_ACGAN_Z_DIM = 128
CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
NUM_CLASSES = len(CIFAR_CLASSES)


# ==============================================================================
# MODEL LOADERS
# ==============================================================================

def load_unconditional_models(device: torch.device):
    """
    Dynamically loads DCGAN Generator and Discriminator from 04a_generative_adversarial_network.py.
    """
    from importlib.machinery import SourceFileLoader

    dcgan_file = os.path.join(SCRIPT_DIR, "04a_generative_adversarial_network.py")
    if not os.path.exists(dcgan_file):
        raise FileNotFoundError(f"Could not find 04a_generative_adversarial_network.py in {SCRIPT_DIR}")

    try:
        mod = SourceFileLoader("dcgan_mod", dcgan_file).load_module()
        netG = mod.Generator32(latent_dim=LATENT_DIM, ngf=64, channels=3)
        netD = mod.Discriminator32(ndf=64, channels=3)
    except Exception as e:
        raise RuntimeError(f"Failed to load DCGAN architecture from {dcgan_file}: {e}")

    if os.path.exists(DCGAN_GEN_CKPT_PATH):
        netG.load_state_dict(torch.load(DCGAN_GEN_CKPT_PATH, map_location=device))
    if os.path.exists(DCGAN_DISC_CKPT_PATH):
        netD.load_state_dict(torch.load(DCGAN_DISC_CKPT_PATH, map_location=device))

    netG.to(device).eval()
    netD.to(device).eval()
    return netG, netD


def load_conditional_models(device: torch.device):
    """
    Dynamically loads Conditional cGAN Generator and Discriminator from 04b_conditional_dcgan.py.
    """
    from importlib.machinery import SourceFileLoader

    cgan_file = os.path.join(SCRIPT_DIR, "04b_conditional_dcgan.py")
    if not os.path.exists(cgan_file):
        raise FileNotFoundError(f"Could not find 04b_conditional_dcgan.py in {SCRIPT_DIR}")

    try:
        mod = SourceFileLoader("cgan_mod", cgan_file).load_module()
        cnetG = mod.ConditionalGenerator32(latent_dim=LATENT_DIM, ngf=64, num_classes=NUM_CLASSES, channels=3)
        cnetD = mod.ConditionalDiscriminator32(ndf=64, num_classes=NUM_CLASSES, channels=3)
    except Exception as e:
        raise RuntimeError(f"Failed to load cGAN architecture from {cgan_file}: {e}")

    if os.path.exists(CGAN_GEN_CKPT_PATH):
        cnetG.load_state_dict(torch.load(CGAN_GEN_CKPT_PATH, map_location=device), strict=False)
    if os.path.exists(CGAN_DISC_CKPT_PATH):
        try:
            cnetD.load_state_dict(torch.load(CGAN_DISC_CKPT_PATH, map_location=device), strict=False)
        except Exception:
            pass

    cnetG.to(device).eval()
    cnetD.to(device).eval()
    return cnetG, cnetD


def load_resnet_acgan_models(device: torch.device):
    """
    Dynamically loads ResNetACGANGenerator and ResNetACGANDiscriminator from 04e_resnet_acgan_cifar10.py.
    """
    from importlib.machinery import SourceFileLoader

    rac_file = os.path.join(SCRIPT_DIR, "04e_resnet_acgan_cifar10.py")
    if not os.path.exists(rac_file):
        raise FileNotFoundError(f"Could not find 04e_resnet_acgan_cifar10.py in {SCRIPT_DIR}")

    try:
        mod = SourceFileLoader("resnet_acgan_mod", rac_file).load_module()
        rac_netG = mod.ResNetACGANGenerator(z_dim=RESNET_ACGAN_Z_DIM, num_classes=NUM_CLASSES, channels=128)
        rac_netG_ema = mod.ResNetACGANGenerator(z_dim=RESNET_ACGAN_Z_DIM, num_classes=NUM_CLASSES, channels=128)
        rac_netD = mod.ResNetACGANDiscriminator(num_classes=NUM_CLASSES, channels=128)
    except Exception as e:
        raise RuntimeError(f"Failed to load ResNet AC-GAN architecture from {rac_file}: {e}")

    if os.path.exists(RESNET_ACGAN_GEN_EMA_CKPT_PATH):
        rac_netG_ema.load_state_dict(torch.load(RESNET_ACGAN_GEN_EMA_CKPT_PATH, map_location=device), strict=False)
    elif os.path.exists(RESNET_ACGAN_GEN_CKPT_PATH):
        rac_netG_ema.load_state_dict(torch.load(RESNET_ACGAN_GEN_CKPT_PATH, map_location=device), strict=False)

    if os.path.exists(RESNET_ACGAN_GEN_CKPT_PATH):
        rac_netG.load_state_dict(torch.load(RESNET_ACGAN_GEN_CKPT_PATH, map_location=device), strict=False)

    if os.path.exists(RESNET_ACGAN_DISC_CKPT_PATH):
        rac_netD.load_state_dict(torch.load(RESNET_ACGAN_DISC_CKPT_PATH, map_location=device), strict=False)

    rac_netG.to(device).eval()
    rac_netG_ema.to(device).eval()
    rac_netD.to(device).eval()
    return rac_netG, rac_netG_ema, rac_netD


def load_rectified_flow_models(device: torch.device):
    """
    Dynamically loads RectifiedFlowUNet and RectifiedFlow from 06_rectified_flow_cifar10.py.
    """
    from importlib.machinery import SourceFileLoader

    rf_file = os.path.join(SCRIPT_DIR, "06_rectified_flow_cifar10.py")
    if not os.path.exists(rf_file):
        raise FileNotFoundError(f"Could not find 06_rectified_flow_cifar10.py in {SCRIPT_DIR}")

    try:
        mod = SourceFileLoader("rf_mod", rf_file).load_module()
        rf_net = mod.RectifiedFlowUNet(in_channels=3, out_channels=3, base_channels=64, num_classes=NUM_CLASSES)
        rf_net_ema = mod.RectifiedFlowUNet(in_channels=3, out_channels=3, base_channels=64, num_classes=NUM_CLASSES)
        rf_handler = mod.RectifiedFlow(time_schedule="uniform", num_classes=NUM_CLASSES)
    except Exception as e:
        raise RuntimeError(f"Failed to load Rectified Flow architecture from {rf_file}: {e}")

    if os.path.exists(RECTIFIED_FLOW_MODEL_EMA_PATH):
        rf_net_ema.load_state_dict(torch.load(RECTIFIED_FLOW_MODEL_EMA_PATH, map_location=device))
    elif os.path.exists(RECTIFIED_FLOW_MODEL_PATH):
        rf_net_ema.load_state_dict(torch.load(RECTIFIED_FLOW_MODEL_PATH, map_location=device))

    if os.path.exists(RECTIFIED_FLOW_MODEL_PATH):
        rf_net.load_state_dict(torch.load(RECTIFIED_FLOW_MODEL_PATH, map_location=device))
    else:
        rf_net.load_state_dict(rf_net_ema.state_dict())

    rf_net.to(device).eval()
    rf_net_ema.to(device).eval()
    return rf_net, rf_net_ema, rf_handler


def load_cddpm_models(device: torch.device):
    """
    Dynamically loads ConditionalUNet and GaussianDiffusion from 05_denoising_diffusion_model.py.
    """
    from importlib.machinery import SourceFileLoader

    ddpm_file = os.path.join(SCRIPT_DIR, "05_denoising_diffusion_model.py")
    if not os.path.exists(ddpm_file):
        return None, None

    try:
        mod = SourceFileLoader("ddpm_mod", ddpm_file).load_module()
        ddpm_net = mod.ConditionalUNet(in_channels=3, out_channels=3, base_channels=48, num_classes=NUM_CLASSES)
        if os.path.exists(CDDPM_MODEL_PATH):
            ddpm_net.load_state_dict(torch.load(CDDPM_MODEL_PATH, map_location=device))
        ddpm_net.to(device).eval()
        diff_handler = mod.GaussianDiffusion(timesteps=1000, beta_schedule="cosine", device=device)
        return ddpm_net, diff_handler
    except Exception as e:
        print(f"Notice: CDDPM loader note: {e}")
        return None, None


def load_real_cifar_dataset() -> Optional[datasets.CIFAR10]:
    """
    Loads test CIFAR-10 samples without downloading if dataset is cached.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    try:
        return datasets.CIFAR10(root=DEFAULT_DATA_DIR, train=False, download=False, transform=transform)
    except Exception:
        return None


def get_milestone_dict(dir_path: str) -> dict:
    """Discovers epoch milestone images from directory sorted by epoch number."""
    if not os.path.exists(dir_path):
        return {}
    res = {}
    for f in sorted(os.listdir(dir_path)):
        if f.startswith("epoch_") and f.endswith(".png"):
            try:
                ep = int(f.replace("epoch_", "").replace(".png", ""))
                res[ep] = f
            except ValueError:
                pass
    return dict(sorted(res.items()))


# ==============================================================================
# MATHEMATICAL HELPERS
# ==============================================================================

def slerp(val: float, low: torch.Tensor, high: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Spherical linear interpolation between two Gaussian noise vectors."""
    low_norm = low / (torch.norm(low, dim=-1, keepdim=True) + eps)
    high_norm = high / (torch.norm(high, dim=-1, keepdim=True) + eps)
    dot = (low_norm * high_norm).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    if torch.abs(so).mean() < eps:
        return (1.0 - val) * low + val * high
    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high


def lerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Linear Euclidean interpolation."""
    return (1.0 - val) * low + val * high


def tensor_to_pil(tensor: torch.Tensor, scale_factor: int = 8) -> Image.Image:
    """Converts a (3, H, W) normalized tensor [-1, 1] to an enlarged crisp PIL Image."""
    t = tensor.detach().cpu().clamp(-1.0, 1.0)
    t = (t + 1.0) / 2.0
    if scale_factor > 1:
        t = F.interpolate(t.unsqueeze(0), scale_factor=scale_factor, mode="nearest").squeeze(0)
    ndarr = t.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    return Image.fromarray(ndarr)


def make_grid_pil(tensors: torch.Tensor, nrow: int = 8, padding: int = 2, scale_factor: Optional[int] = None) -> Image.Image:
    """Creates a grid of images from a (B, 3, H, W) tensor [-1, 1] with adaptive zoom."""
    t = tensors.detach().cpu().clamp(-1.0, 1.0)
    t = (t + 1.0) / 2.0
    b = t.shape[0]

    # Adaptive scaling so all outputs fill the large display panel nicely
    if scale_factor is None:
        if b <= 4:
            scale_factor = 8   # e.g. 2x2 grid -> 512x512
        elif b <= 16:
            scale_factor = 4   # e.g. 4x4 grid -> 512x512
        elif b <= 64:
            scale_factor = 3   # e.g. 8x8 grid -> 768x768
        else:
            scale_factor = 2   # 100+ images -> 640x640

    if scale_factor > 1:
        t = F.interpolate(t, scale_factor=scale_factor, mode="nearest")

    pad_val = max(2, scale_factor)
    grid = torchvision.utils.make_grid(t, nrow=nrow, padding=pad_val, pad_value=0.1)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    return Image.fromarray(ndarr)


# ==============================================================================
# GRADIO APPLICATION BUILDER
# ==============================================================================

def build_app(device_str: Optional[str] = None):
    device = torch.device(device_str if device_str else get_device())
    print(f"Loading Unified Generative Models on device: {device}...")

    # Load All Models
    dcgan_netG, dcgan_netD = load_unconditional_models(device)
    cgan_netG, cgan_netD = load_conditional_models(device)
    rac_netG, rac_netG_ema, rac_netD = load_resnet_acgan_models(device)
    rf_net, rf_net_ema, rf_handler = load_rectified_flow_models(device)
    cddpm_net, diff_handler = load_cddpm_models(device)
    cifar_testset = load_real_cifar_dataset()

    # Discover Milestone Snapshots
    dcgan_milestone_files = get_milestone_dict(DCGAN_MILESTONES_DIR)
    cgan_milestone_files = get_milestone_dict(CGAN_MILESTONES_DIR)
    rac_milestone_files = get_milestone_dict(RESNET_ACGAN_MILESTONES_DIR)
    rf_milestone_files = get_milestone_dict(RECTIFIED_FLOW_MILESTONES_DIR)
    cddpm_milestone_files = get_milestone_dict(CDDPM_MILESTONES_DIR)

    # --------------------------------------------------------------------------
    # TAB 1 LOGIC: UNCONDITIONAL DCGAN
    # --------------------------------------------------------------------------
    def run_unconditional_synthesis(batch_size: int, seed: int):
        batch_size = int(batch_size)
        nrow = int(math.ceil(math.sqrt(batch_size)))
        if seed is not None and seed >= 0:
            torch.manual_seed(int(seed))
        t0 = time.time()
        z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
        with torch.no_grad():
            fakes = dcgan_netG(z)
            scores = torch.sigmoid(dcgan_netD(fakes)).view(-1).cpu().numpy()
        elapsed = time.time() - t0
        pil_grid = make_grid_pil(fakes, nrow=nrow, padding=2)
        stats = (
            f"### 📊 Synthesis Report: Unconditional DCGAN\n"
            f"- **Model**: Deep Convolutional GAN (Radford et al. 2016)\n"
            f"- **Conditioning**: None (Unguided isotropic $\\mathbf{{z}} \\sim \\mathcal{{N}}(\\mathbf{{0}}, \\mathbf{{I}}_{{100}})$)\n"
            f"- **Batch Size**: `{batch_size}` samples ({nrow}x{nrow} grid) | **Seed**: `{seed}`\n"
            f"- **Inference Time**: **{elapsed:.3f}s** ({elapsed / batch_size * 1000:.1f} ms/sample on `{device}`)\n"
            f"- **Discriminator Realness $D(G(z))$**: Mean = **{scores.mean():.4f}**, Range = `[{scores.min():.4f}, {scores.max():.4f}]`\n"
            f"- **Key Architecture**: Transposed convolutions (`ConvTranspose2d`) stride 2 $\\implies$ subject to periodic checkerboard artifacts."
        )
        return pil_grid, stats

    def run_unconditional_benchmark_matrix(seed: int):
        return run_unconditional_synthesis(batch_size=64, seed=seed)

    def run_slerp_interpolation(interp_type: str, steps: int, seed_a: int, seed_b: int):
        steps = int(steps)
        torch.manual_seed(int(seed_a))
        z_a = torch.randn(1, LATENT_DIM, device=device)
        torch.manual_seed(int(seed_b))
        z_b = torch.randn(1, LATENT_DIM, device=device)

        alphas = np.linspace(0.0, 1.0, steps)
        z_traj = []
        for a in alphas:
            z_t = slerp(float(a), z_a, z_b) if interp_type == "Spherical (Slerp)" else lerp(float(a), z_a, z_b)
            z_traj.append(z_t)
        z_batch = torch.cat(z_traj, dim=0).view(steps, LATENT_DIM, 1, 1)
        with torch.no_grad():
            fakes = dcgan_netG(z_batch)
        pil_grid = make_grid_pil(fakes, nrow=steps, padding=2)
        desc = (
            f"**Latent Traversal ($z_A \\to z_B$)**:\n"
            f"- Method: `{interp_type}` over {steps} steps\n"
            f"- Seed A: `{seed_a}`, Seed B: `{seed_b}`\n"
            f"- Visual inspection reveals smooth semantic transitions across the learned data manifold."
        )
        return pil_grid, desc

    def run_latent_sweep(base_seed: int, target_dim: int, sweep_range: float, num_steps: int):
        num_steps = int(num_steps)
        dim_idx = int(target_dim) % LATENT_DIM
        torch.manual_seed(int(base_seed))
        z_base = torch.randn(1, LATENT_DIM, device=device)

        vals = np.linspace(-float(sweep_range), float(sweep_range), num_steps)
        z_list = []
        for v in vals:
            z_mod = z_base.clone()
            z_mod[0, dim_idx] = float(v)
            z_list.append(z_mod)
        z_batch = torch.cat(z_list, dim=0).view(num_steps, LATENT_DIM, 1, 1)
        with torch.no_grad():
            fakes = dcgan_netG(z_batch)
        pil_grid = make_grid_pil(fakes, nrow=num_steps, padding=2)
        desc = (
            f"**Coordinate Sweep on Latent Dimension $z_{{{dim_idx}}}$**:\n"
            f"- Coordinate Range: `[-{sweep_range}, +{sweep_range}]` across `{num_steps}` steps\n"
            f"- Demonstrates semantic disentanglement along axis $z_{{{dim_idx}}}$."
        )
        return pil_grid, desc

    # --------------------------------------------------------------------------
    # TAB 2 LOGIC: CONDITIONAL cGAN
    # --------------------------------------------------------------------------
    def run_cgan_synthesis(class_name: str, batch_size: int, seed: int):
        if class_name == "All 10 Classes (10x10 Matrix)":
            return generate_cgan_class_matrix(seed)
        batch_size = int(batch_size)
        nrow = int(math.ceil(math.sqrt(batch_size)))

        cls_idx = CIFAR_CLASSES.index(class_name)
        if seed is not None and seed >= 0:
            torch.manual_seed(int(seed))

        t0 = time.time()
        z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
        labels = torch.full((batch_size,), cls_idx, dtype=torch.long, device=device)

        with torch.no_grad():
            fakes = cgan_netG(z, labels)
            d_out = cgan_netD(fakes, labels)
            scores = torch.sigmoid(d_out).view(-1).cpu().numpy()

        elapsed = time.time() - t0
        pil_grid = make_grid_pil(fakes, nrow=nrow, padding=2)
        desc = (
            f"### 📊 Synthesis Report: Class-Conditional cGAN\n"
            f"- **Model**: Class-Conditional DCGAN (Mirza & Osindero 2014)\n"
            f"- **Conditioning**: Targeted Class **{class_name.upper()}** (Index `{cls_idx}`)\n"
            f"- **Batch Size**: `{batch_size}` samples ({nrow}x{nrow} grid) | **Seed**: `{seed}`\n"
            f"- **Inference Time**: **{elapsed:.3f}s** ({elapsed / batch_size * 1000:.1f} ms/sample on `{device}`)\n"
            f"- **Conditioned Realness $D(x, y)$**: Mean = **{scores.mean():.4f}**, Range = `[{scores.min():.4f}, {scores.max():.4f}]`\n"
            f"- **Key Architecture**: 100-D label embedding concatenated into noise vector $\\mathbf{{z}}$."
        )
        return pil_grid, desc

    def generate_cgan_class_matrix(seed: int):
        if seed is not None and seed >= 0:
            torch.manual_seed(int(seed))
        num_per_class = 10
        total_samples = NUM_CLASSES * num_per_class

        t0 = time.time()
        z = torch.randn(total_samples, LATENT_DIM, 1, 1, device=device)
        labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(num_per_class)

        with torch.no_grad():
            fakes = cgan_netG(z, labels)

        elapsed = time.time() - t0
        pil_grid = make_grid_pil(fakes, nrow=num_per_class, padding=2)
        desc = (
            f"### 📊 Synthesis Report: cGAN 10x10 Class Matrix\n"
            f"- **Model**: Class-Conditional DCGAN (Mirza & Osindero 2014)\n"
            f"- **Conditioning**: All 10 CIFAR-10 Classes (10 images per row, 100 samples total)\n"
            f"- **Seed**: `{seed}` | **Inference Time**: **{elapsed:.3f}s** ({elapsed / total_samples * 1000:.1f} ms/sample on `{device}`)\n"
            f"- **Observations**: Class conditioning guides general colors and coarse blobs, but intra-class diversity and edge definition remain limited."
        )
        return pil_grid, desc

    def evaluate_cgan_discriminator(source: str, test_class: str, sample_idx: int):
        cls_idx = CIFAR_CLASSES.index(test_class)
        if source == "Real CIFAR-10 Test Sample":
            if cifar_testset is None:
                return None, "CIFAR-10 dataset not cached locally."
            idx = int(sample_idx) % len(cifar_testset)
            img_t, actual_label = cifar_testset[idx]
            img_t = img_t.unsqueeze(0).to(device)
            status_src = f"Real Test Image #{idx} (Ground Truth: **{CIFAR_CLASSES[actual_label]}**)"
        else:
            torch.manual_seed(int(sample_idx))
            z = torch.randn(1, LATENT_DIM, 1, 1, device=device)
            gen_label = torch.tensor([cls_idx], device=device)
            with torch.no_grad():
                img_t = cgan_netG(z, gen_label)
            status_src = f"cGAN Generated Fake (Conditioned on **{test_class}**)"

        eval_label = torch.tensor([cls_idx], device=device)
        with torch.no_grad():
            d_raw = cgan_netD(img_t, eval_label)
            score = torch.sigmoid(d_raw).item()

        pil_img = tensor_to_pil(img_t[0])
        report = (
            f"**Discriminator Verification**:\n"
            f"- Source: {status_src}\n"
            f"- Condition Tested: **{test_class}** (Class `{cls_idx}`)\n"
            f"- Realness Probability $P(\\text{{Real}} \\mid x, y) = \\mathbf{{{score * 100:.2f}\\%}}$\n"
            f"- Discriminator Logit: `{d_raw.item():.4f}`\n"
            f"- Verdict: {'✅ Strong Real Match' if score >= 0.5 else '⚠️ Flagged as Fake or Label Mismatch'}"
        )
        return pil_img, report

    # --------------------------------------------------------------------------
    # TAB 3 LOGIC: MODERN RESNET AC-GAN
    # --------------------------------------------------------------------------
    def run_resnet_acgan_synthesis(class_name: str, batch_size: int, use_ema: bool, seed: int):
        if class_name == "All 10 Classes (10x10 Matrix)":
            return generate_resnet_acgan_class_matrix(use_ema, seed)
        batch_size = int(batch_size)
        nrow = int(math.ceil(math.sqrt(batch_size)))

        cls_idx = CIFAR_CLASSES.index(class_name)
        if seed is not None and seed >= 0:
            torch.manual_seed(int(seed))

        t0 = time.time()
        z = torch.randn(batch_size, RESNET_ACGAN_Z_DIM, device=device)
        labels = torch.full((batch_size,), cls_idx, dtype=torch.long, device=device)

        active_netG = rac_netG_ema if use_ema else rac_netG
        with torch.no_grad():
            fakes = active_netG(z, labels)
            adv_logits, cls_logits = rac_netD(fakes)
            real_scores = torch.sigmoid(adv_logits).view(-1).cpu().numpy()
            cls_probs = F.softmax(cls_logits, dim=-1)
            target_probs = cls_probs[:, cls_idx].cpu().numpy()
            pred_classes = cls_probs.argmax(dim=-1).cpu().numpy()
            accuracy = (pred_classes == cls_idx).mean() * 100.0

        elapsed = time.time() - t0
        pil_grid = make_grid_pil(fakes, nrow=nrow, padding=2)
        desc = (
            f"### 📊 Synthesis Report: Modern ResNet AC-GAN\n"
            f"- **Model**: ResNet AC-GAN with Spectral Normalization (Odena et al. ICML 2017)\n"
            f"- **Conditioning**: Targeted Class **{class_name.upper()}** (Index `{cls_idx}`) with Conditional Batch Norm\n"
            f"- **Weights**: `{'EMA (Exponential Moving Average)' if use_ema else 'Raw Training Weights'}`\n"
            f"- **Batch Size**: `{batch_size}` samples ({nrow}x{nrow} grid) | **Seed**: `{seed}`\n"
            f"- **Inference Time**: **{elapsed:.3f}s** ({elapsed / batch_size * 1000:.1f} ms/sample on `{device}`)\n"
            f"- **Realness $D_{{\\text{{adv}}}}(x)$**: Mean = **{real_scores.mean():.4f}** | **Aux Classifier Accuracy $D_{{\\text{{cls}}}}$**: **{accuracy:.1f}%** (Confidence: `{target_probs.mean() * 100:.1f}%`)\n"
            f"- **Key Architecture**: Nearest-neighbor 2x upsampling completely eliminates checkerboard artifacts."
        )
        return pil_grid, desc

    def generate_resnet_acgan_class_matrix(use_ema: bool, seed: int):
        if seed is not None and seed >= 0:
            torch.manual_seed(int(seed))
        num_per_class = 10
        total_samples = NUM_CLASSES * num_per_class

        t0 = time.time()
        z = torch.randn(total_samples, RESNET_ACGAN_Z_DIM, device=device)
        labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(num_per_class)

        active_netG = rac_netG_ema if use_ema else rac_netG
        with torch.no_grad():
            fakes = active_netG(z, labels)
            _, cls_logits = rac_netD(fakes)
            preds = cls_logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item() * 100.0

        elapsed = time.time() - t0
        pil_grid = make_grid_pil(fakes, nrow=num_per_class, padding=2)
        desc = (
            f"### 📊 Synthesis Report: ResNet AC-GAN 10x10 Class Matrix\n"
            f"- **Model**: Modern ResNet AC-GAN (Odena et al. ICML 2017)\n"
            f"- **Conditioning**: All 10 CIFAR-10 Classes (10 samples per row, 100 samples total)\n"
            f"- **Weights**: `{'EMA Smoothed' if use_ema else 'Raw'}` | **Seed**: `{seed}`\n"
            f"- **Inference Time**: **{elapsed:.3f}s** ({elapsed / total_samples * 1000:.1f} ms/sample on `{device}`)\n"
            f"- **Auxiliary Classifier Alignment**: **{acc:.1f}%** of synthesized samples correctly identified by $D_{{\\text{{cls}}}}$."
        )
        return pil_grid, desc

    def evaluate_resnet_acgan_discriminator(source: str, test_class: str, sample_idx: int, use_ema: bool):
        cls_idx = CIFAR_CLASSES.index(test_class)
        if source == "Real CIFAR-10 Test Sample":
            if cifar_testset is None:
                return None, "CIFAR-10 dataset not cached locally."
            idx = int(sample_idx) % len(cifar_testset)
            img_t, actual_label = cifar_testset[idx]
            img_t = img_t.unsqueeze(0).to(device)
            status_src = f"Real Test Image #{idx} (Ground Truth: **{CIFAR_CLASSES[actual_label]}**)"
        else:
            torch.manual_seed(int(sample_idx))
            z = torch.randn(1, RESNET_ACGAN_Z_DIM, device=device)
            active_netG = rac_netG_ema if use_ema else rac_netG
            with torch.no_grad():
                img_t = active_netG(z, torch.tensor([cls_idx], device=device))
            status_src = f"ResNet AC-GAN Fake (Conditioned on **{test_class}**, {'EMA' if use_ema else 'Raw'})"

        with torch.no_grad():
            adv_logit, cls_logits = rac_netD(img_t)
            real_prob = torch.sigmoid(adv_logit).item()
            cls_probs = F.softmax(cls_logits, dim=-1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(cls_probs))

        pil_img = tensor_to_pil(img_t[0])
        top_probs = "\n".join([
            f"  - {CIFAR_CLASSES[i]}: `{cls_probs[i]*100:.2f}%` {'👈 (Top)' if i == pred_idx else ''}"
            for i in np.argsort(-cls_probs)[:3]
        ])

        report = (
            f"**Dual-Head Discriminator Output**:\n"
            f"- Source: {status_src}\n"
            f"- Adversarial Realness $D_{{\\text{{adv}}}}(x)$: **{real_prob * 100:.2f}%** (`logit: {adv_logit.item():.4f}`)\n"
            f"- Auxiliary Classification $D_{{\\text{{cls}}}}(x)$ Prediction: **{CIFAR_CLASSES[pred_idx].upper()}** ({cls_probs[pred_idx]*100:.2f}% confidence)\n"
            f"- Top Class Probabilities:\n{top_probs}\n"
            f"- Status: {'✅ Correct Alignment' if pred_idx == cls_idx else '⚠️ Class Disagreement'}"
        )
        return pil_img, report

    # --------------------------------------------------------------------------
    # TAB 4 LOGIC: DIFFUSION & RECTIFIED FLOW
    # --------------------------------------------------------------------------
    def generate_flow_class_matrix(
        model_choice: str,
        steps: int,
        solver: str,
        cfg_scale: float,
        use_ema: bool,
        seed: int,
    ):
        steps = int(steps)
        cfg_scale = float(cfg_scale)
        seed = int(seed)
        if seed >= 0:
            torch.manual_seed(seed)

        num_per_class = 10
        total_samples = NUM_CLASSES * num_per_class
        labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(num_per_class)

        t_start = time.time()
        if "Rectified Flow" in model_choice:
            active_model = rf_net_ema if use_ema else rf_net
            with torch.no_grad():
                samples = rf_handler.sample_ode(
                    active_model,
                    shape=(total_samples, 3, 32, 32),
                    classes=labels,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    solver=solver,
                    device=device,
                )
        else:
            if cddpm_net is None or diff_handler is None:
                return None, "Conditional DDPM model or weights not loaded."
            with torch.no_grad():
                if solver == "ddpm":
                    samples = diff_handler.sample_ddpm(cddpm_net, labels=labels, steps=steps, cfg_scale=cfg_scale)
                else:
                    samples = diff_handler.sample_ddim(cddpm_net, labels=labels, ddim_steps=steps, cfg_scale=cfg_scale)

        elapsed = time.time() - t_start
        pil_grid = make_grid_pil(samples, nrow=num_per_class, padding=2)
        desc = (
            f"### 📊 Synthesis Report: Flow / Diffusion 10x10 Matrix\n"
            f"- **Model**: `{model_choice}`\n"
            f"- **Conditioning**: All 10 CIFAR-10 Classes (10 samples per row, 100 samples total)\n"
            f"- **Solver**: `{solver.upper()}` with `{steps}` steps | **CFG Scale**: `{cfg_scale}`\n"
            f"- **Weights**: `{'EMA Smoothed' if use_ema else 'Raw'}` | **Seed**: `{seed}`\n"
            f"- **Inference Time**: **{elapsed:.2f}s** ({elapsed / total_samples * 1000:.1f} ms/sample on `{device}`)\n"
            f"- **Key Architecture**: Continuous velocity ODE integration $\\frac{{d\\mathbf{{x}}_t}}{{dt}} = v_\\theta(\\mathbf{{x}}_t, t, y)$."
        )
        return pil_grid, desc

    def run_flow_diffusion_synthesis(
        model_choice: str,
        target_class_str: str,
        steps: int,
        solver: str,
        cfg_scale: float,
        use_ema: bool,
        seed: int,
        batch_size: int,
    ):
        if target_class_str == "All 10 Classes (10x10 Matrix)":
            return generate_flow_class_matrix(model_choice, steps, solver, cfg_scale, use_ema, seed)

        steps = int(steps)
        cfg_scale = float(cfg_scale)
        seed = int(seed)
        batch_size = int(batch_size)

        if seed >= 0:
            torch.manual_seed(seed)

        t_start = time.time()

        if "Rectified Flow" in model_choice:
            active_model = rf_net_ema if use_ema else rf_net
            cls_idx = CIFAR_CLASSES.index(target_class_str)
            labels = torch.full((batch_size,), cls_idx, dtype=torch.long, device=device)
            b = batch_size
            nrow = int(math.ceil(math.sqrt(batch_size)))

            shape = (b, 3, 32, 32)
            with torch.no_grad():
                samples = rf_handler.sample_ode(
                    active_model,
                    shape=shape,
                    classes=labels,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    solver=solver,
                    device=device,
                )
            elapsed = time.time() - t_start
            nfe = steps * (2 if cfg_scale > 1.0 else 1) * (2 if solver == "heun" else 1)

            pil_grid = make_grid_pil(samples, nrow=nrow, padding=2)
            desc = (
                f"### 📊 Synthesis Report: Rectified Flow (Flow Matching)\n"
                f"- **Model**: 1-Rectified Flow (Liu et al. 2022 / InstaFlow)\n"
                f"- **Conditioning**: Targeted Class **{target_class_str.upper()}** (Index `{cls_idx}`)\n"
                f"- **Weights**: `{'EMA (Exponential Moving Average)' if use_ema else 'Raw Training Weights'}`\n"
                f"- **Solver**: `{solver.upper()}` with `{steps}` integration steps | **CFG Scale**: `{cfg_scale}`\n"
                f"- **Batch Size**: `{b}` samples ({nrow}x{nrow} grid) | **Seed**: `{seed}`\n"
                f"- **Inference Time**: **{elapsed:.2f}s** ({elapsed / b * 1000:.1f} ms/sample on `{device}`) | **NFE**: `{nfe}`\n"
                f"- **Key Architecture**: Velocity field $v_\\theta$ follows straight ODE transport lines $\\implies$ high fidelity in 15–25 steps!"
            )
            return pil_grid, desc

        else:
            # CDDPM / DDIM
            if cddpm_net is None or diff_handler is None:
                return None, "Conditional DDPM model or weights not loaded."

            cls_idx = CIFAR_CLASSES.index(target_class_str)
            labels = torch.full((batch_size,), cls_idx, dtype=torch.long, device=device)
            b = batch_size
            nrow = int(math.ceil(math.sqrt(batch_size)))

            with torch.no_grad():
                if solver == "ddpm":
                    samples = diff_handler.sample_ddpm(cddpm_net, labels=labels, steps=steps, cfg_scale=cfg_scale)
                else:
                    samples = diff_handler.sample_ddim(cddpm_net, labels=labels, ddim_steps=steps, cfg_scale=cfg_scale)

            elapsed = time.time() - t_start
            pil_grid = make_grid_pil(samples, nrow=nrow, padding=2)
            desc = (
                f"### 📊 Synthesis Report: Conditional Diffusion (DDPM / DDIM)\n"
                f"- **Model**: Conditional ResNet U-Net (Ho et al. 2020; Song et al. 2021)\n"
                f"- **Conditioning**: Targeted Class **{target_class_str.upper()}** (Index `{cls_idx}`)\n"
                f"- **Sampling**: `{solver.upper()}` with `{steps}` discretization steps | **CFG Scale**: `{cfg_scale}`\n"
                f"- **Batch Size**: `{b}` samples ({nrow}x{nrow} grid) | **Seed**: `{seed}`\n"
                f"- **Inference Time**: **{elapsed:.2f}s** ({elapsed / b * 1000:.1f} ms/sample on `{device}`)\n"
                f"- **Key Architecture**: Score-based reverse diffusion with cosine noise schedule."
            )
            return pil_grid, desc

    def run_flow_trajectory_visualization(
        model_choice: str,
        target_class_str: str,
        steps: int,
        solver: str,
        cfg_scale: float,
        use_ema: bool,
        seed: int,
    ):
        steps = int(steps)
        cfg_scale = float(cfg_scale)
        seed = int(seed)

        if seed >= 0:
            torch.manual_seed(seed)

        cls_idx = 0 if target_class_str == "All 10 Classes (Grid)" else CIFAR_CLASSES.index(target_class_str)
        cls_name = CIFAR_CLASSES[cls_idx]
        b = 1
        shape = (b, 3, 32, 32)
        classes = torch.tensor([cls_idx], device=device)

        if "Rectified Flow" in model_choice:
            active_model = rf_net_ema if use_ema else rf_net
            x = torch.randn(shape, device=device)
            timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)
            null_classes = torch.full((b,), NUM_CLASSES, device=device, dtype=torch.long)

            num_keyframes = 6
            frame_indices = set(torch.linspace(0, steps, num_keyframes).round().long().tolist())
            frames = []

            with torch.no_grad():
                for i in range(steps):
                    if i in frame_indices:
                        frames.append(x.clone().clamp(-1.0, 1.0))
                    t_curr = timesteps[i]
                    t_next = timesteps[i + 1]
                    dt = t_curr - t_next
                    t_batch = torch.full((b,), t_curr.item(), device=device)

                    if cfg_scale > 1.0:
                        v_cond = active_model(x, t_batch, classes)
                        v_uncond = active_model(x, t_batch, null_classes)
                        v = v_uncond + cfg_scale * (v_cond - v_uncond)
                    else:
                        v = active_model(x, t_batch, classes)

                    if solver == "heun" and i < steps - 1:
                        x_pred = x - dt * v
                        t_next_batch = torch.full((b,), t_next.item(), device=device)
                        if cfg_scale > 1.0:
                            v_next_c = active_model(x_pred, t_next_batch, classes)
                            v_next_u = active_model(x_pred, t_next_batch, null_classes)
                            v_next = v_next_u + cfg_scale * (v_next_c - v_next_u)
                        else:
                            v_next = active_model(x_pred, t_next_batch, classes)
                        x = x - 0.5 * dt * (v + v_next)
                    else:
                        x = x - dt * v

                frames.append(x.clone().clamp(-1.0, 1.0))

            cat_frames = torch.cat(frames, dim=0)
            pil_filmstrip = make_grid_pil(cat_frames, nrow=len(frames), padding=4, scale_factor=3)
            desc = (
                f"**Continuous Reverse ODE Flow Trajectory Filmstrip**:\n"
                f"- Target Class: **{cls_name.upper()}** (Index `{cls_idx}`)\n"
                f"- Evolution: Pure Gaussian Noise at $t=1.0$ $\\longrightarrow$ Clean Synthesized Image at $t=0.0$\n"
                f"- Keyframes Displayed: `{len(frames)}` intermediate integration states ($t \\approx 1.0, 0.8, 0.6, 0.4, 0.2, 0.0$)\n"
                f"- Notice the linear velocity transport: high-level semantic contours crystallize first, followed by fine textures!"
            )
            return pil_filmstrip, desc

        else:
            # DDIM Trajectory
            if cddpm_net is None or diff_handler is None:
                return None, "Conditional DDPM model or weights not loaded."

            x = torch.randn(shape, device=device)
            c = diff_handler.timesteps // steps
            time_steps = torch.arange(0, diff_handler.timesteps, c, device=device).long()
            if time_steps[-1] != diff_handler.timesteps - 1:
                time_steps = torch.cat([time_steps, torch.tensor([diff_handler.timesteps - 1], device=device)])
            time_steps_prev = torch.cat([torch.tensor([-1], device=device), time_steps[:-1]])

            null_labels = torch.full_like(classes, NUM_CLASSES)
            frames = [x.clone().clamp(-1.0, 1.0)]
            save_interval = max(1, len(time_steps) // 5)

            with torch.no_grad():
                for idx, i in enumerate(reversed(range(len(time_steps)))):
                    t_val = time_steps[i]
                    t_prev_val = time_steps_prev[i]
                    t = torch.full((b,), t_val, device=device, dtype=torch.long)

                    if cfg_scale > 1.0:
                        x_in = torch.cat([x, x], dim=0)
                        t_in = torch.cat([t, t], dim=0)
                        y_in = torch.cat([classes, null_labels], dim=0)
                        eps_both = cddpm_net(x_in, t_in, y_in)
                        eps_cond, eps_uncond = eps_both.chunk(2, dim=0)
                        eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
                    else:
                        eps_pred = cddpm_net(x, t, classes)

                    alpha_hat_t = diff_handler.alpha_hat[t_val]
                    alpha_hat_prev = diff_handler.alpha_hat[t_prev_val] if t_prev_val >= 0 else torch.tensor(1.0, device=device)
                    pred_x0 = (x - torch.sqrt(1.0 - alpha_hat_t) * eps_pred) / torch.sqrt(alpha_hat_t)
                    pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
                    pred_dir = torch.sqrt(torch.clamp(1.0 - alpha_hat_prev, min=0.0)) * eps_pred
                    x = torch.sqrt(alpha_hat_prev) * pred_x0 + pred_dir

                    if (idx + 1) % save_interval == 0 or idx == len(time_steps) - 1:
                        frames.append(x.clone().clamp(-1.0, 1.0))

            cat_frames = torch.cat(frames, dim=0)
            pil_filmstrip = make_grid_pil(cat_frames, nrow=len(frames), padding=4, scale_factor=3)
            desc = (
                f"**DDIM Reverse Denoising Trajectory Filmstrip**:\n"
                f"- Target Class: **{cls_name.upper()}**\n"
                f"- Evolution: Pure Gaussian Noise $T=1000$ $\\longrightarrow$ Denoised Output $T=0$\n"
                f"- Intermediate Keyframes: `{len(frames)}` progressive denoising stages."
            )
            return pil_filmstrip, desc

    def load_precomputed_sheet(choice: str):
        if "Heun 25" in choice and os.path.exists(RECTIFIED_FLOW_SAMPLE_GRID_PATH):
            return Image.open(RECTIFIED_FLOW_SAMPLE_GRID_PATH)
        elif "Raw Weights" in choice and os.path.exists(RECTIFIED_FLOW_SAMPLE_RAW_PATH):
            return Image.open(RECTIFIED_FLOW_SAMPLE_RAW_PATH)
        elif "DDPM" in choice and os.path.exists(CDDPM_SAMPLE_GRID_PATH):
            return Image.open(CDDPM_SAMPLE_GRID_PATH)
        elif os.path.exists(RECTIFIED_FLOW_SAMPLE_GRID_PATH):
            return Image.open(RECTIFIED_FLOW_SAMPLE_GRID_PATH)
        return None

    # --------------------------------------------------------------------------
    # TAB 5 LOGIC: TRAINING MILESTONES
    # --------------------------------------------------------------------------
    def load_dcgan_milestone(ep: int):
        fname = dcgan_milestone_files.get(int(ep), "epoch_30.png")
        fpath = os.path.join(DCGAN_MILESTONES_DIR, fname)
        return Image.open(fpath) if os.path.exists(fpath) else None

    def load_cgan_milestone(ep: int):
        fname = cgan_milestone_files.get(int(ep), "epoch_50.png")
        fpath = os.path.join(CGAN_MILESTONES_DIR, fname)
        return Image.open(fpath) if os.path.exists(fpath) else None

    def load_rac_milestone(ep: int):
        fname = rac_milestone_files.get(int(ep), "epoch_100.png")
        fpath = os.path.join(RESNET_ACGAN_MILESTONES_DIR, fname)
        return Image.open(fpath) if os.path.exists(fpath) else None

    def load_rf_milestone(ep: int):
        fname = rf_milestone_files.get(int(ep), "epoch_50.png")
        fpath = os.path.join(RECTIFIED_FLOW_MILESTONES_DIR, fname)
        return Image.open(fpath) if os.path.exists(fpath) else None

    # ==========================================================================
    # GRADIO USER INTERFACE LAYOUT
    # ==========================================================================

    custom_css = """
    .gradio-container { max-width: 1450px !important; margin: auto; }
    .header-box { text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px; margin-bottom: 1.5rem; color: white; }
    .header-box h1 { margin-bottom: 0.3rem; font-size: 2.2rem; font-weight: 800; }
    .header-box p { margin: 0; opacity: 0.85; font-size: 1.05rem; }
    .large-display-image { min-height: 520px !important; }
    .large-display-image img, .large-display-image .image-container img {
        image-rendering: -webkit-optimize-contrast;
        image-rendering: pixelated;
        image-rendering: crisp-edges;
        max-height: 600px !important;
        object-fit: contain !important;
        margin: auto;
    }
    .mile-display-image { min-height: 420px !important; }
    .mile-display-image img {
        image-rendering: pixelated;
        max-height: 440px !important;
        object-fit: contain !important;
    }
    """

    with gr.Blocks(title="IF5152: Generative Models Studio", css=custom_css, theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_classes=["header-box"]):
            gr.Markdown(
                """
                # 🎨 IF5152: Generative Models Studio
                ### Comprehensive CIFAR-10 Generative AI: DCGAN • Conditional cGAN • ResNet AC-GAN • Flow Matching & Diffusion
                *Institut Teknologi Bandung — Advanced Machine Learning & Generative Models*
                """
            )

        with gr.Tabs() as main_tabs:

            # ------------------------------------------------------------------
            # TAB 1: UNCONDITIONAL DCGAN
            # ------------------------------------------------------------------
            with gr.Tab("🎲 Tab 1: Unconditional DCGAN", id="tab_unconditional"):
                gr.Markdown(
                    """
                    ### 🎲 Deep Convolutional Generative Adversarial Network (Radford et al., ICLR 2016)
                    The unconditional DCGAN maps a 100-dimensional isotropic Gaussian vector $\\mathbf{z} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})$
                    into a $3 \\times 32 \\times 32$ CIFAR-10 image using transposed convolutions.
                    """
                )
                with gr.Tabs():
                    with gr.TabItem("🎨 Interactive Synthesis"):
                        with gr.Row():
                            with gr.Column(scale=4):
                                dcgan_class = gr.Dropdown(
                                    choices=["Unconditional (Random / No Label)"],
                                    value="Unconditional (Random / No Label)",
                                    label="Target Semantic Class",
                                    interactive=False
                                )
                                dcgan_batch = gr.Slider(minimum=1, maximum=16, value=4, step=1, label="Batch Size (Single Class)")
                                dcgan_seed = gr.Number(value=42, label="Random Seed (-1 for Random)", precision=0)
                                with gr.Row():
                                    btn_dcgan_gen = gr.Button("🎨 Generate Samples", variant="primary")
                                    btn_dcgan_matrix = gr.Button("🌟 Generate 8x8 Benchmark Grid (64 Samples)", variant="secondary")
                            with gr.Column(scale=8):
                                dcgan_out_img = gr.Image(label="Synthesized Visual Output", type="pil", height=540, elem_classes=["large-display-image"])
                                dcgan_stats_md = gr.Markdown()
                        btn_dcgan_gen.click(run_unconditional_synthesis, inputs=[dcgan_batch, dcgan_seed], outputs=[dcgan_out_img, dcgan_stats_md])
                        btn_dcgan_matrix.click(run_unconditional_benchmark_matrix, inputs=[dcgan_seed], outputs=[dcgan_out_img, dcgan_stats_md])

                    with gr.TabItem("🌐 Latent Interpolation (Slerp vs Lerp)"):
                        with gr.Row():
                            with gr.Column(scale=4):
                                interp_type = gr.Radio(choices=["Spherical (Slerp)", "Euclidean (Lerp)"], value="Spherical (Slerp)", label="Interpolation Method")
                                interp_steps = gr.Slider(minimum=5, maximum=16, value=8, step=1, label="Interpolation Steps")
                                with gr.Row():
                                    seed_a = gr.Number(value=101, label="Seed A", precision=0)
                                    seed_b = gr.Number(value=202, label="Seed B", precision=0)
                                btn_interp = gr.Button("🧭 Interpolate Latent Trajectory", variant="primary")
                            with gr.Column(scale=8):
                                interp_out_img = gr.Image(label="Continuous Latent Manifold Walk", type="pil", height=460, elem_classes=["large-display-image"])
                                interp_stats_md = gr.Markdown()
                        btn_interp.click(run_slerp_interpolation, inputs=[interp_type, interp_steps, seed_a, seed_b], outputs=[interp_out_img, interp_stats_md])

                    with gr.TabItem("🎚️ Latent Coordinate Sweep"):
                        with gr.Row():
                            with gr.Column(scale=4):
                                sweep_dim = gr.Slider(minimum=0, maximum=99, value=7, step=1, label="Target Latent Coordinate (z_i)")
                                sweep_range = gr.Slider(minimum=1.0, maximum=4.0, value=2.5, step=0.5, label="Sweep Range [-R, +R]")
                                sweep_steps = gr.Slider(minimum=5, maximum=15, value=9, step=1, label="Sweep Steps")
                                sweep_seed = gr.Number(value=42, label="Base Anchor Seed", precision=0)
                                btn_sweep = gr.Button("🔍 Sweep Latent Coordinate", variant="primary")
                            with gr.Column(scale=8):
                                sweep_out_img = gr.Image(label="Coordinate Sweep Trajectory", type="pil", height=460, elem_classes=["large-display-image"])
                                sweep_stats_md = gr.Markdown()
                        btn_sweep.click(run_latent_sweep, inputs=[sweep_seed, sweep_dim, sweep_range, sweep_steps], outputs=[sweep_out_img, sweep_stats_md])

            # ------------------------------------------------------------------
            # TAB 2: CONDITIONAL cGAN
            # ------------------------------------------------------------------
            with gr.Tab("🎯 Tab 2: Class-Conditional cGAN", id="tab_conditional"):
                gr.Markdown(
                    """
                    ### 🎯 Class-Conditional DCGAN (Mirza & Osindero, 2014)
                    Guiding generation by providing semantic label $y \\in \\{0, \\dots, 9\\}$.
                    - **Generator**: 100-D label embedding is concatenated with $\\mathbf{z} \\in \\mathbb{R}^{100}$.
                    - **Discriminator**: Spatial label channel ($1 \\times 32 \\times 32$) is concatenated with input image ($4 \\times 32 \\times 32$).
                    """
                )
                with gr.Tabs():
                    with gr.TabItem("🎨 Interactive Synthesis"):
                        with gr.Row():
                            with gr.Column(scale=4):
                                cgan_class = gr.Dropdown(
                                    choices=["All 10 Classes (10x10 Matrix)"] + CIFAR_CLASSES,
                                    value="airplane",
                                    label="Target Semantic Class"
                                )
                                cgan_batch = gr.Slider(minimum=1, maximum=16, value=4, step=1, label="Batch Size (Single Class)")
                                cgan_seed = gr.Number(value=42, label="Random Seed (-1 for Random)", precision=0)
                                with gr.Row():
                                    btn_cgan_gen = gr.Button("🎨 Generate Samples", variant="primary")
                                    btn_cgan_matrix = gr.Button("🌟 Generate 10x10 Class Matrix (100 Samples)", variant="secondary")
                            with gr.Column(scale=8):
                                cgan_out_img = gr.Image(label="Synthesized Visual Output", type="pil", height=540, elem_classes=["large-display-image"])
                                cgan_stats_md = gr.Markdown()
                        btn_cgan_gen.click(run_cgan_synthesis, inputs=[cgan_class, cgan_batch, cgan_seed], outputs=[cgan_out_img, cgan_stats_md])
                        btn_cgan_matrix.click(generate_cgan_class_matrix, inputs=[cgan_seed], outputs=[cgan_out_img, cgan_stats_md])

                    with gr.TabItem("🕵️ Conditional Discriminator Audit"):
                        with gr.Row():
                            with gr.Column(scale=4):
                                d_audit_source = gr.Radio(choices=["Generated cGAN Fake", "Real CIFAR-10 Test Sample"], value="Generated cGAN Fake", label="Sample Source")
                                d_audit_class = gr.Dropdown(choices=CIFAR_CLASSES, value="frog", label="Tested Class Condition")
                                d_audit_idx = gr.Number(value=7, label="Sample Index / Seed", precision=0)
                                btn_d_audit = gr.Button("🔬 Inspect Discriminator Verification", variant="primary")
                            with gr.Column(scale=8):
                                d_audit_img = gr.Image(label="Evaluated Image", type="pil", height=480, elem_classes=["large-display-image"])
                                d_audit_report = gr.Markdown()
                        btn_d_audit.click(evaluate_cgan_discriminator, inputs=[d_audit_source, d_audit_class, d_audit_idx], outputs=[d_audit_img, d_audit_report])

            # ------------------------------------------------------------------
            # TAB 3: MODERN RESNET AC-GAN
            # ------------------------------------------------------------------
            with gr.Tab("🏗️ Tab 3: ResNet AC-GAN", id="tab_resnet_acgan"):
                gr.Markdown(
                    """
                    ### 🏗️ Modern ResNet AC-GAN (Auxiliary Classifier GAN) (Odena et al., ICML 2017)
                    Modern architectural enhancements over vanilla DCGAN:
                    - **Checkerboard-Free Upsampling**: Nearest-Neighbor 2x + $3 \\times 3$ Convolutions (replaces ConvTranspose2d).
                    - **Conditional Batch Normalization (CBN)**: Modulates feature activations dynamically per class.
                    - **Dual-Head Discriminator**: Separates adversarial realness $D_{\\text{adv}}(x)$ from classification $D_{\\text{cls}}(x)$.
                    - **Spectral Normalization**: Enforces 1-Lipschitz continuity $\\sigma(W) \\le 1$ on all discriminator layers.
                    - **Exponential Moving Average (EMA)**: Smooths generator weights with $\\beta = 0.999$.
                    """
                )
                with gr.Tabs():
                    with gr.TabItem("🎨 Interactive Synthesis"):
                        with gr.Row():
                            with gr.Column(scale=4):
                                rac_class = gr.Dropdown(
                                    choices=["All 10 Classes (10x10 Matrix)"] + CIFAR_CLASSES,
                                    value="airplane",
                                    label="Target Semantic Class"
                                )
                                rac_batch = gr.Slider(minimum=1, maximum=16, value=4, step=1, label="Batch Size (Single Class)")
                                rac_seed = gr.Number(value=42, label="Random Seed (-1 for Random)", precision=0)
                                rac_use_ema = gr.Checkbox(value=True, label="Use EMA Smoothed Weights")
                                with gr.Row():
                                    btn_rac_gen = gr.Button("🎨 Generate Samples", variant="primary")
                                    btn_rac_matrix = gr.Button("🌟 Generate 10x10 Class Matrix (100 Samples)", variant="secondary")
                            with gr.Column(scale=8):
                                rac_out_img = gr.Image(label="Synthesized Visual Output", type="pil", height=540, elem_classes=["large-display-image"])
                                rac_stats_md = gr.Markdown()
                        btn_rac_gen.click(run_resnet_acgan_synthesis, inputs=[rac_class, rac_batch, rac_use_ema, rac_seed], outputs=[rac_out_img, rac_stats_md])
                        btn_rac_matrix.click(generate_resnet_acgan_class_matrix, inputs=[rac_use_ema, rac_seed], outputs=[rac_out_img, rac_stats_md])

                    with gr.TabItem("🕵️ Dual-Head Discriminator Inspection"):
                        with gr.Row():
                            with gr.Column(scale=4):
                                rac_d_source = gr.Radio(choices=["Generated ResNet AC-GAN Fake", "Real CIFAR-10 Test Sample"], value="Generated ResNet AC-GAN Fake", label="Sample Source")
                                rac_d_class = gr.Dropdown(choices=CIFAR_CLASSES, value="automobile", label="Target Semantic Class")
                                rac_d_idx = gr.Number(value=10, label="Sample Index / Seed", precision=0)
                                rac_d_ema = gr.Checkbox(value=True, label="Use EMA Generator")
                                btn_rac_d_audit = gr.Button("🔬 Run Dual-Head Diagnostic", variant="primary")
                            with gr.Column(scale=8):
                                rac_d_img = gr.Image(label="Evaluated Image", type="pil", height=480, elem_classes=["large-display-image"])
                                rac_d_report = gr.Markdown()
                        btn_rac_d_audit.click(evaluate_resnet_acgan_discriminator, inputs=[rac_d_source, rac_d_class, rac_d_idx, rac_d_ema], outputs=[rac_d_img, rac_d_report])

            # ------------------------------------------------------------------
            # TAB 4: DIFFUSION & RECTIFIED FLOW
            # ------------------------------------------------------------------
            with gr.Tab("🌊 Tab 4: Diffusion & Rectified Flow", id="tab_diffusion_flow"):
                gr.Markdown(
                    """
                    ### 🌊 Continuous-Time Generative Models: Flow Matching & Denoising Diffusion
                    Moving beyond minimax adversarial games (GANs), continuous-time generative models learn to invert a forward noise-corrupting trajectory.
                    - **Rectified Flow (1-RF)** (*Liu et al., 2022; Lipman et al., 2023*): Learns straight-line velocity vectors $v_\\theta(\\mathbf{x}_t, t, y) \\approx \\mathbf{x}_1 - \\mathbf{x}_0$ bridging Gaussian noise $\\mathbf{x}_1$ directly to clean data $\\mathbf{x}_0$. Because velocity paths are straight lines, ODE solvers like **Euler** and **Heun** generate high-fidelity samples in as few as **10–25 steps** without curvature error!
                    - **Conditional DDPM / DDIM** (*Ho et al., 2020; Song et al., 2021*): Learns score function $\\nabla_x \\log p_t(x)$ via noise prediction $\\epsilon_\\theta(\\mathbf{x}_t, t, y)$ with cosine variance schedule. Deterministically inverted with DDIM.
                    """
                )
                with gr.Tabs():
                    with gr.TabItem("🎨 Interactive Synthesis"):
                        with gr.Row():
                            with gr.Column(scale=4):
                                rf_model_choice = gr.Radio(
                                    choices=["Rectified Flow (1-RF Flow Matching)", "Conditional DDPM / DDIM"],
                                    value="Rectified Flow (1-RF Flow Matching)",
                                    label="Model Architecture"
                                )
                                rf_target_class = gr.Dropdown(
                                    choices=["All 10 Classes (10x10 Matrix)"] + CIFAR_CLASSES,
                                    value="airplane",
                                    label="Target Semantic Class"
                                )
                                rf_batch_slider = gr.Slider(minimum=1, maximum=16, value=4, step=1, label="Batch Size (Single Class)")
                                rf_seed_num = gr.Number(value=42, label="Random Seed (-1 for Random)", precision=0)
                                with gr.Row():
                                    rf_steps_slider = gr.Slider(minimum=5, maximum=50, value=20, step=5, label="Sampling / ODE Steps (N)")
                                    rf_solver_choice = gr.Dropdown(choices=["euler", "heun"], value="heun", label="ODE / SDE Solver")
                                with gr.Row():
                                    rf_cfg_slider = gr.Slider(minimum=1.0, maximum=4.0, value=1.5, step=0.1, label="CFG Guidance Scale (w)")
                                    rf_ema_toggle = gr.Checkbox(value=True, label="Use EMA Weights")

                                with gr.Row():
                                    btn_rf_gen = gr.Button("🎨 Generate Samples", variant="primary")
                                    btn_rf_matrix = gr.Button("🌟 Generate 10x10 Class Matrix (100 Samples)", variant="secondary")

                            with gr.Column(scale=8):
                                rf_out_img = gr.Image(label="Synthesized Visual Output", type="pil", height=540, elem_classes=["large-display-image"])
                                rf_status_md = gr.Markdown()

                        # Dynamic solver dropdown update on model radio change
                        def on_model_choice_change(m):
                            if "Rectified Flow" in m:
                                return gr.Dropdown(choices=["euler", "heun"], value="heun")
                            else:
                                return gr.Dropdown(choices=["ddim", "ddpm"], value="ddim")

                        rf_model_choice.change(on_model_choice_change, inputs=rf_model_choice, outputs=rf_solver_choice)

                        btn_rf_gen.click(
                            run_flow_diffusion_synthesis,
                            inputs=[rf_model_choice, rf_target_class, rf_steps_slider, rf_solver_choice, rf_cfg_slider, rf_ema_toggle, rf_seed_num, rf_batch_slider],
                            outputs=[rf_out_img, rf_status_md]
                        )
                        btn_rf_matrix.click(
                            generate_flow_class_matrix,
                            inputs=[rf_model_choice, rf_steps_slider, rf_solver_choice, rf_cfg_slider, rf_ema_toggle, rf_seed_num],
                            outputs=[rf_out_img, rf_status_md]
                        )

                    with gr.TabItem("🎞️ Visualize Flow / Denoising Trajectory"):
                        with gr.Row():
                            with gr.Column(scale=4):
                                traj_model_choice = gr.Radio(
                                    choices=["Rectified Flow (1-RF Flow Matching)", "Conditional DDPM / DDIM"],
                                    value="Rectified Flow (1-RF Flow Matching)",
                                    label="Model Architecture"
                                )
                                traj_class = gr.Dropdown(choices=CIFAR_CLASSES, value="automobile", label="Target Class for Trajectory")
                                traj_steps = gr.Slider(minimum=5, maximum=50, value=20, step=5, label="Trajectory Steps (N)")
                                traj_solver = gr.Dropdown(choices=["euler", "heun"], value="heun", label="Solver")
                                traj_cfg = gr.Slider(minimum=1.0, maximum=4.0, value=1.5, step=0.1, label="CFG Guidance Scale")
                                traj_seed = gr.Number(value=42, label="Seed", precision=0)
                                traj_ema = gr.Checkbox(value=True, label="Use EMA Weights")
                                btn_run_traj = gr.Button("🎞️ Run Reverse Trajectory Simulation", variant="primary")
                            with gr.Column(scale=8):
                                traj_out_img = gr.Image(label="Continuous Reverse Trajectory Filmstrip (t=1.0 -> 0.0)", type="pil", height=420, elem_classes=["large-display-image"])
                                traj_status_md = gr.Markdown()

                        traj_model_choice.change(on_model_choice_change, inputs=traj_model_choice, outputs=traj_solver)
                        btn_run_traj.click(
                            run_flow_trajectory_visualization,
                            inputs=[traj_model_choice, traj_class, traj_steps, traj_solver, traj_cfg, traj_ema, traj_seed],
                            outputs=[traj_out_img, traj_status_md]
                        )

                    with gr.TabItem("🖼️ Pre-Computed Benchmark Evaluation Sheets"):
                        gr.Markdown(
                            """
                            View official evaluation sheets rendered directly from the trained checkpoints on disk ($10 \\times 10 = 100$ samples spanning all CIFAR-10 classes).
                            """
                        )
                        with gr.Row():
                            sheet_choice = gr.Radio(
                                choices=[
                                    "Rectified Flow (EMA, Heun 25 steps, Epoch 50/150)",
                                    "Rectified Flow (Raw Weights, Euler 50 steps)",
                                    "Conditional DDPM (Cosine Schedule, Epoch 10)",
                                ],
                                value="Rectified Flow (EMA, Heun 25 steps, Epoch 50/150)",
                                label="Evaluation Sheet"
                            )
                        img_eval_sheet = gr.Image(label="Full Benchmark 10x10 Evaluation Grid", type="pil", height=650, elem_classes=["large-display-image"])
                        sheet_choice.change(load_precomputed_sheet, inputs=sheet_choice, outputs=img_eval_sheet)

            # ------------------------------------------------------------------
            # TAB 5: TRAINING MILESTONES & GENERATIVE TAXONOMY
            # ------------------------------------------------------------------
            with gr.Tab("📈 Tab 5: Training Milestones & Generative Taxonomy", id="tab_milestones"):
                gr.Markdown(
                    """
                    ### 📈 Epoch-by-Epoch Evolution Across Generative Paradigms
                    Select milestone checkpoints to trace the visual evolution from initial noise to structured semantic objects.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("#### 1. Unconditional DCGAN")
                        dcgan_epoch_sel = gr.Dropdown(choices=list(dcgan_milestone_files.keys()), value=30 if 30 in dcgan_milestone_files else None, label="DCGAN Epoch")
                        img_dcgan_mile = gr.Image(label="DCGAN Snapshot", type="pil", height=420, elem_classes=["mile-display-image"])
                    with gr.Column(scale=3):
                        gr.Markdown("#### 2. Conditional DCGAN")
                        cgan_epoch_sel = gr.Dropdown(choices=list(cgan_milestone_files.keys()), value=50 if 50 in cgan_milestone_files else None, label="cGAN Epoch")
                        img_cgan_mile = gr.Image(label="cGAN Snapshot", type="pil", height=420, elem_classes=["mile-display-image"])
                    with gr.Column(scale=3):
                        gr.Markdown("#### 3. Modern ResNet AC-GAN")
                        rac_epoch_sel = gr.Dropdown(choices=list(rac_milestone_files.keys()), value=100 if 100 in rac_milestone_files else None, label="ResNet AC-GAN Epoch")
                        img_rac_mile = gr.Image(label="ResNet AC-GAN Snapshot", type="pil", height=420, elem_classes=["mile-display-image"])
                    with gr.Column(scale=3):
                        gr.Markdown("#### 4. Rectified Flow (1-RF)")
                        rf_epoch_sel = gr.Dropdown(choices=list(rf_milestone_files.keys()), value=50 if 50 in rf_milestone_files else (150 if 150 in rf_milestone_files else None), label="Rectified Flow Epoch")
                        img_rf_mile = gr.Image(label="Rectified Flow Snapshot", type="pil", height=420, elem_classes=["mile-display-image"])

                dcgan_epoch_sel.change(load_dcgan_milestone, inputs=dcgan_epoch_sel, outputs=img_dcgan_mile)
                cgan_epoch_sel.change(load_cgan_milestone, inputs=cgan_epoch_sel, outputs=img_cgan_mile)
                rac_epoch_sel.change(load_rac_milestone, inputs=rac_epoch_sel, outputs=img_rac_mile)
                rf_epoch_sel.change(load_rf_milestone, inputs=rf_epoch_sel, outputs=img_rf_mile)

                demo.load(
                    lambda: (
                        load_dcgan_milestone(30),
                        load_cgan_milestone(50),
                        load_rac_milestone(100),
                        load_rf_milestone(50),
                        load_precomputed_sheet("Rectified Flow (EMA, Heun 25 steps, Epoch 50/150)"),
                    ),
                    outputs=[img_dcgan_mile, img_cgan_mile, img_rac_mile, img_rf_mile, img_eval_sheet]
                )

                gr.Markdown(
                    """
                    ---
                    ### 🏗️ Theoretical Comparison: Generative Models Taxonomy
                    | Paradigm | Model | Generator Mapping | Conditioning | Loss Formulation | Inference Steps (NFE) | Mode Coverage / Stability |
                    | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
                    | **Adversarial (GAN)** | **DCGAN** (Radford 2016) | $\\mathbf{z} \\in \\mathbb{R}^{100} \\to \\mathbf{x}$ | None (Unguided) | Minimax JS Divergence $\\min_G \\max_D$ | **1** | Prone to mode collapse; checkerboard artifacts |
                    | **Adversarial (cGAN)** | **Conditional DCGAN** (Mirza 2014) | $(\\mathbf{z}, y) \\to \\mathbf{x}$ | Latent concatenation + label channel | Conditional Minimax GAN Loss | **1** | Mode collapse risk; unguided spatial grid |
                    | **Adversarial (ResNet)** | **ResNet AC-GAN** (Odena 2017) | $(\\mathbf{z}, y) \\to \\mathbf{x}$ | **Conditional Batch Norm (CBN)** | Dual Head: Realness + Auxiliary Classifier | **1** | **High stability** (Spectral Norm $\\sigma(W) \\le 1$ + EMA) |
                    | **Diffusion (Score)** | **DDPM / DDIM** (Ho 2020, Song 2021) | $\\mathbf{x}_T \\sim \\mathcal{N}(0, I) \\to \\mathbf{x}_0$ | Sinusoidal Time + Class Embedding (CFG) | Denoising Score Matching $\\mathbb{E}[\\|\\epsilon - \\epsilon_\\theta\\|^2]$ | **20 – 100** | **Total mode coverage**; curved trajectories |
                    | **Flow Matching (CNF)** | **Rectified Flow (1-RF)** (Liu 2022) | $\\mathbf{x}_1 \\sim \\mathcal{N}(0, I) \\to \\mathbf{x}_0$ | Sinusoidal Time + Class Embedding (CFG) | Straight Velocity $\\mathbb{E}[\\| (\\mathbf{x}_1 - \\mathbf{x}_0) - v_\\theta\\|^2]$ | **10 – 25** | **Optimal transport**; straight ODE paths; fast sampling |

                    ---
                    ### 🔬 Why Do Continuous-Time Generative Models (Flow & Diffusion) Outperform GANs on CIFAR-10?
                    1. **Elimination of the Minimax Instability**:
                       - GANs rely on an adversarial zero-sum game $\\min_G \\max_D V(D, G)$. If $D$ overpowers $G$, gradients vanish; if $D$ is too weak, mode collapse occurs.
                       - Diffusion and Flow Matching optimize a **pure mean-squared error regression loss** ($\\mathcal{L}_{\\text{MSE}}$). There is no adversary, ensuring monotonic convergence.
                    2. **Straight Velocity Field vs. Curved Brownian Motion**:
                       - Standard diffusion models introduce Brownian noise at each step, creating tortuous curved sampling trajectories that require 50–1000 steps.
                       - **Rectified Flow** connects noise $\\mathbf{x}_1$ directly to data $\\mathbf{x}_0$ via straight lines $\\mathbf{x}_t = t\\mathbf{x}_1 + (1-t)\\mathbf{x}_0$. Because velocity $v_\\theta = \\mathbf{x}_1 - \\mathbf{x}_0$ is constant along trajectories, 2nd-order solvers like **Heun** achieve exceptional sharpness in just 20–25 steps!
                    3. **Architectural Superiority**:
                       - Uses deep U-Nets with **Self-Attention blocks** at $16 \\times 16$ resolution and **Classifier-Free Guidance (CFG)** to boost semantic alignment and vivid color saturation.
                    """
                )

        gr.Markdown(
            """
            ---
            📖 **Technical Guides & Implementations**:
            - [Guide 04: Generative Adversarial Networks (DCGAN & cGAN)](docs/04_generative_adversarial_networks.md)
            - [Guide 05: Denoising Diffusion Probabilistic Models (DDPM & DDIM)](docs/05_diffusion_probabilistic_models.md)
            - [Guide 06: Flow Matching & Rectified Flow](docs/06_rectified_flow.md)
            - Python Implementations: [04a DCGAN](04a_generative_adversarial_network.py) | [04b cGAN](04b_conditional_dcgan.py) | [04e ResNet AC-GAN](04e_resnet_acgan_cifar10.py) | [05 DDPM](05_denoising_diffusion_model.py) | [06 Rectified Flow](06_rectified_flow_cifar10.py)
            """
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generative Models Studio (CIFAR-10)")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on ('cpu', 'cuda', 'mps')")
    parser.add_argument("--server-port", type=int, default=7860, help="Port to run the Gradio server on")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    demo = build_app(device_str=args.device)
    demo.launch(server_port=args.server_port, share=args.share)
