"""
========================================================================================
IF5152: Video Understanding - Session 3 Lab
File: 03_video_transformer_inference.py
========================================================================================

PEDAGOGICAL & THEORETICAL FOUNDATIONS:
----------------------------------------------------------------------------------------
1. Self-Attention in Spatio-Temporal Space:
   - 2D Convolutions have a local receptive field. To capture interactions between distant
     regions in space and time, early methods used Non-Local Blocks or 3D CNNs.
   - Vision Transformers (ViT) compute global self-attention across all tokens.

2. From 2D ViT to Video Vision Transformers:
   - ViViT (Video Vision Transformer; Arnab et al., ICCV 2021):
     Extends 2D ViT to video. Model 2 (The Factorized Encoder) processes each frame with
     a shared 2D ViT backbone, extracting spatial [CLS] tokens, which are then passed
     through a lightweight temporal Transformer encoder (O(T^2)).
     -> King of transfer learning: 100% of 2D ViT weights can be reused directly!
   - VideoMAE (Tong et al., NeurIPS 2022):
     Pretrained with self-supervised Masked Autoencoding at an ultra-high 90% masking ratio.
     Uses spatiotemporal tubelet embedding (2x16x16) and joint space-time self-attention.

3. Industrial Perspective: Foundation Models vs Training from Scratch:
   - Training Video Transformers from scratch requires massive compute clusters (e.g. 64+ A100s).
   - In industry and modern research, engineers either fine-tune ViViT/TimeSformer via transfer
     learning or deploy foundation models like VideoMAE pretrained on Kinetics-400.

USAGE:
  # 1. Run default VideoMAE inference (16 frames)
  python 03_video_transformer_inference.py --video data/person_bicycle.mp4

  # 2. Run ViViT inference (32 frames)
  python 03_video_transformer_inference.py --model vivit --device mps

  # 3. Head-to-Head Comparative Benchmark (VideoMAE vs ViViT)
  python 03_video_transformer_inference.py --model all --device mps

  # 4. Inspect Educational ViViT Factorized Architecture
  python 03_video_transformer_inference.py --print-model
========================================================================================
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Ensure local lab directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_device, print_device_info, read_video_pyav, resolve_data_path

# Hugging Face Transformers
from transformers import (
    AutoImageProcessor,
    VideoMAEForVideoClassification,
    VivitForVideoClassification,
    VivitImageProcessor,
)


# ======================================================================================
# EDUCATIONAL PYTORCH IMPLEMENTATION: ViViT Model 2 (Factorized Encoder)
# ======================================================================================
class ViViTFactorizedModel(nn.Module):
    """
    Educational reference implementation of ViViT Model 2 (The Factorized Encoder).
    Reference: Arnab et al., "ViViT: A Video Vision Transformer", ICCV 2021.

    Architectural Blueprint:
      1. Spatial ViT Encoder (Shared across all T frames):
         Extracts spatial patch tokens (196 tokens) + spatial [CLS] token for each frame.
         The spatial [CLS] vector z_t in R^D serves as the frame summary representation.
      2. Temporal Transformer Encoder:
         Sequence of T summary tokens [z_1, ..., z_T] is prepended with a learnable
         Temporal [CLS] token and injected with temporal position embeddings.
         Processes inter-frame dynamics with self-attention across time (O(T^2)).
      3. Classification Head:
         Projects final Temporal [CLS] token to Kinetics-400 action logits.

    Why this design is ideal for fine-tuning & transfer learning:
      - The spatial encoder can be directly initialized from any ImageNet-pretrained ViT
        (e.g., google/vit-base-patch16-224 or torchvision.models.vit_b_16) with ZERO weight surgery!
      - Feature caching: Freeze the spatial backbone, extract [T, D] embeddings, and train
        only the lightweight temporal encoder in minutes on a single GPU.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        temporal_depth: int = 4,
        temporal_heads: int = 8,
        num_classes: int = 400,
        max_frames: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_frames = max_frames

        # Spatial Encoder: 2D Vision Transformer backbone
        try:
            import torchvision.models as models
            vit = models.vit_b_16(weights=None)
            vit.heads = nn.Identity()  # Remove classification head to output embed_dim (768)
            self.spatial_encoder = vit
        except Exception:
            # Fallback lightweight spatial backbone for environments without torchvision
            self.spatial_encoder = nn.Sequential(
                nn.Conv2d(3, embed_dim, kernel_size=16, stride=16),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )

        # Temporal [CLS] token and temporal positional embeddings
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, max_frames + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        # Temporal Transformer: Computes self-attention purely across the time axis (O(T^2))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=temporal_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=temporal_depth)
        self.norm = nn.LayerNorm(embed_dim)

        # Final action classification head
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: Input video tensor of shape [Batch, Time, Channels, Height, Width]
        Returns:
          logits: Action classification logits of shape [Batch, num_classes]
        """
        B, T, C, H, W = x.shape
        if T > self.max_frames:
            raise ValueError(f"Input clip has {T} frames, but max_frames is {self.max_frames}")

        # Step 1: Flatten batch and time to pass frames through 2D spatial backbone
        x_flat = x.view(B * T, C, H, W)
        z_spatial = self.spatial_encoder(x_flat)  # [B * T, embed_dim]
        z_spatial = z_spatial.view(B, T, self.embed_dim)  # [B, T, embed_dim]

        # Step 2: Prepend learnable Temporal [CLS] token
        cls_tokens = self.temporal_cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        z_seq = torch.cat((cls_tokens, z_spatial), dim=1)  # [B, T + 1, embed_dim]

        # Step 3: Add temporal positional embeddings
        z_seq = z_seq + self.temporal_pos_embed[:, : T + 1, :]
        z_seq = self.pos_drop(z_seq)

        # Step 4: Temporal Self-Attention across frames
        out = self.temporal_transformer(z_seq)  # [B, T + 1, embed_dim]
        out = self.norm(out)

        # Step 5: Extract final Temporal [CLS] representation and classify
        temporal_cls = out[:, 0]  # [B, embed_dim]
        logits = self.classifier(temporal_cls)  # [B, num_classes]
        return logits


# ======================================================================================
# KINETICS-400 LABEL DICTIONARY RESOLVER
# ======================================================================================
def get_kinetics400_labels(videomae_model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics") -> Dict[int, str]:
    """
    Retrieves human-readable action names for all 400 Kinetics classes.
    Uses VideoMAE's validated config as reference to ensure standard class ordering.
    """
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(videomae_model_name)
        return {int(k): v for k, v in cfg.id2label.items()}
    except Exception:
        # Fallback dummy map if offline
        return {i: f"class_{i}" for i in range(400)}


# ======================================================================================
# INFERENCE RUNNERS
# ======================================================================================
def run_videomae_inference(
    raw_frames: np.ndarray,
    device: torch.device,
    model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
    id2label: Dict[int, str] = None,
) -> Tuple[torch.Tensor, float, int]:
    """
    Runs VideoMAE inference on a clip of 16 frames.
    Returns: (probabilities [400], latency_ms, total_params)
    """
    print(f"\n--- Loading VideoMAE Preprocessor & Weights: '{model_name}' ---")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(model_name).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"VideoMAE Parameters : {total_params:,}")
    print(f"Input Frames        : {len(raw_frames)} frames -> Tubelet (2x16x16) -> 1,568 tokens")

    inputs = processor(list(raw_frames), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        t0 = time.perf_counter()
        outputs = model(**inputs)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = torch.softmax(outputs.logits[0], dim=-1)
    return probs, latency_ms, total_params


def run_vivit_inference(
    raw_frames: np.ndarray,
    device: torch.device,
    model_name: str = "google/vivit-b-16x2-kinetics400",
    id2label: Dict[int, str] = None,
) -> Tuple[torch.Tensor, float, int]:
    """
    Runs ViViT inference on a clip of 32 frames.
    Returns: (probabilities [400], latency_ms, total_params)
    """
    print(f"\n--- Loading ViViT Preprocessor & Weights: '{model_name}' ---")
    processor = VivitImageProcessor.from_pretrained(model_name)
    model = VivitForVideoClassification.from_pretrained(model_name).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"ViViT Parameters    : {total_params:,}")
    print(f"Input Frames        : {len(raw_frames)} frames -> Tubelet (2x16x16) -> 3,136 tokens")

    inputs = processor(list(raw_frames), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        t0 = time.perf_counter()
        outputs = model(**inputs)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = torch.softmax(outputs.logits[0], dim=-1)
    return probs, latency_ms, total_params


def display_top5(probs: torch.Tensor, id2label: Dict[int, str], title: str = "TOP-5 PREDICTIONS"):
    """Formats and prints top-5 ranked action predictions with an ASCII bar chart."""
    top5_probs, top5_indices = torch.topk(probs, k=5)
    print("\n" + "=" * 56)
    print(f" {title}")
    print("=" * 56)
    for rank, (p, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
        label_name = id2label.get(idx.item(), f"class_{idx.item()}")
        bar = "█" * int(p.item() * 35)
        print(f"  #{rank}  {label_name:<28} : {p.item() * 100:5.2f}%  {bar}")
    print("=" * 56)


# ======================================================================================
# MAIN EXECUTION ENTRYPOINT
# ======================================================================================
def main():
    parser = argparse.ArgumentParser(
        description="IF5152 Lecture 11 Lab: Pretrained Video Transformers (VideoMAE & ViViT)"
    )
    parser.add_argument("--video", type=str, default="data/person_bicycle.mp4",
                        help="Path to input video clip (MP4 format)")
    parser.add_argument("--model", type=str, default="videomae",
                        choices=["videomae", "vivit", "all", "custom"],
                        help="Model selection: 'videomae' | 'vivit' | 'all' (head-to-head comparison)")
    parser.add_argument("--custom_checkpoint", type=str, default=None,
                        help="Optional Hugging Face repo ID for custom transformer checkpoint")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Number of uniformly sampled frames (defaults: VideoMAE=16, ViViT=32)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Target device: 'auto' | 'mps' | 'cuda' | 'cpu'")
    parser.add_argument("--print-model", action="store_true",
                        help="Print the PyTorch educational ViViT factorized architecture and exit")
    args = parser.parse_args()

    # If --print-model is flagged, print educational ViViT architecture and exit
    if args.print_model:
        print("\n" + "=" * 70)
        print(" EDUCATIONAL PYTORCH IMPLEMENTATION: ViViT MODEL 2 (FACTORIZED ENCODER)")
        print("=" * 70)
        edu_model = ViViTFactorizedModel()
        print(edu_model)
        tot = sum(p.numel() for p in edu_model.parameters())
        spatial_p = sum(p.numel() for p in edu_model.spatial_encoder.parameters())
        temporal_p = sum(p.numel() for p in edu_model.temporal_transformer.parameters())
        print("-" * 70)
        print(f"Total Parameters     : {tot:,}")
        print(f"Spatial Backbone     : {spatial_p:,} params (Directly transferable from 2D ViT!)")
        print(f"Temporal Transformer : {temporal_p:,} params (Lightweight inter-frame self-attention)")
        print("=" * 70)
        return

    # 1. Hardware Inspection & Video Path Resolution
    device = get_device(args.device)
    print_device_info(device)
    video_path = resolve_data_path(args.video)

    # 2. Resolve Kinetics-400 Action Labels
    k400_id2label = get_kinetics400_labels()

    # 3. Model Execution Logic
    if args.model == "videomae":
        frames_to_decode = args.num_frames or 16
        print(f"\nDecoding {frames_to_decode} frames for VideoMAE from '{video_path}'...")
        raw_frames = read_video_pyav(video_path, num_frames=frames_to_decode, fast_seek=True)
        ckpt = args.custom_checkpoint or "MCG-NJU/videomae-base-finetuned-kinetics"
        probs, latency_ms, _ = run_videomae_inference(raw_frames, device, ckpt, k400_id2label)
        display_top5(probs, k400_id2label, "TOP-5 PREDICTIONS (VideoMAE)")
        print(f"\nInference Latency: {latency_ms:.2f} ms on {device.type.upper()}")

    elif args.model == "vivit":
        frames_to_decode = args.num_frames or 32
        print(f"\nDecoding {frames_to_decode} frames for ViViT from '{video_path}'...")
        raw_frames = read_video_pyav(video_path, num_frames=frames_to_decode, fast_seek=True)
        ckpt = args.custom_checkpoint or "google/vivit-b-16x2-kinetics400"
        probs, latency_ms, _ = run_vivit_inference(raw_frames, device, ckpt, k400_id2label)
        display_top5(probs, k400_id2label, "TOP-5 PREDICTIONS (ViViT)")
        print(f"\nInference Latency: {latency_ms:.2f} ms on {device.type.upper()}")

    elif args.model == "all":
        print("\n" + "=" * 70)
        print(" HEAD-TO-HEAD BENCHMARK: VideoMAE vs ViViT")
        print("=" * 70)
        print(f"Decoding 32 frames from '{video_path}' for synchronized evaluation...")
        raw_frames_32 = read_video_pyav(video_path, num_frames=32, fast_seek=True)
        raw_frames_16 = raw_frames_32[::2]  # Subsample 16 frames spanning the identical temporal window

        # Run VideoMAE
        v_probs, v_ms, v_params = run_videomae_inference(
            raw_frames_16, device, "MCG-NJU/videomae-base-finetuned-kinetics", k400_id2label
        )
        display_top5(v_probs, k400_id2label, "VIDEOMAE TOP-5 PREDICTIONS")

        # Run ViViT
        vivit_probs, vivit_ms, vivit_params = run_vivit_inference(
            raw_frames_32, device, "google/vivit-b-16x2-kinetics400", k400_id2label
        )
        display_top5(vivit_probs, k400_id2label, "VIVIT TOP-5 PREDICTIONS")

        # Top-1 extractions
        v_top1_idx = torch.argmax(v_probs).item()
        v_top1_lbl = k400_id2label.get(v_top1_idx, f"class_{v_top1_idx}")
        v_top1_conf = v_probs[v_top1_idx].item() * 100

        vivit_top1_idx = torch.argmax(vivit_probs).item()
        vivit_top1_lbl = k400_id2label.get(vivit_top1_idx, f"class_{vivit_top1_idx}")
        vivit_top1_conf = vivit_probs[vivit_top1_idx].item() * 100

        # Comparative Summary Table
        print("\n" + "=" * 76)
        print(f" {'ARCHITECTURAL & PERFORMANCE COMPARISON TABLE':^74} ")
        print("=" * 76)
        header = f" {'Metric / Property':<24} | {'VideoMAE (Tong 2022)':<23} | {'ViViT (Arnab 2021)':<22}"
        print(header)
        print("-" * 76)
        print(f" {'Input Frames (T)':<24} | {16:<23} | {32:<22}")
        print(f" {'Tubelet Size':<24} | {'2 x 16 x 16':<23} | {'2 x 16 x 16':<22}")
        print(f" {'Spatiotemporal Tokens':<24} | {'1,568 tokens':<23} | {'3,136 tokens':<22}")
        print(f" {'Total Parameters':<24} | {f'{v_params:,}':<23} | {f'{vivit_params:,}':<22}")
        print(f" {'Inference Latency':<24} | {f'{v_ms:.1f} ms':<23} | {f'{vivit_ms:.1f} ms':<22}")
        print(f" {'Top-1 Predicted Class':<24} | {v_top1_lbl:<23} | {vivit_top1_lbl:<22}")
        print(f" {'Top-1 Confidence':<24} | {f'{v_top1_conf:.2f}%':<23} | {f'{vivit_top1_conf:.2f}%':<22}")
        print("=" * 76)

        # Pedagogical Takeaway
        latency_ratio = vivit_ms / max(v_ms, 1e-5)
        print(f"\n[Pedagogical Takeaway for Students]")
        print(f"  - ViViT processes 32 frames (3,136 tokens), resulting in a ~{latency_ratio:.1f}x latency scaling")
        print(f"    over VideoMAE's 16 frames (1,568 tokens), demonstrating the quadratic O(N^2) token attention cost.")
        print(f"  - ViViT provides seamless transfer learning from 2D ViT weights, whereas VideoMAE")
        print(f"    is optimized through large-scale self-supervised masked pretraining at 90% masking ratio.")

    print("\n[SUCCESS] Session 3 (Video Transformer) lab execution completed.")


if __name__ == "__main__":
    main()
