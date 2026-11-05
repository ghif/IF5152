"""
========================================================================================
IF5152: Video Understanding - Session 5 Lab
File: 05_spatiotemporal_3d_cnns.py
========================================================================================

PEDAGOGICAL & THEORETICAL FOUNDATIONS:
----------------------------------------------------------------------------------------
1. From 2D Late Fusion to 3D Spatio-Temporal Convolutions:
   - In 2D Late Fusion (Session 1), spatial features are extracted independently per frame
     using a 2D CNN, and time is aggregated only at the final bottleneck layer.
   - In 3D Convolutions, kernels slide across both Space (H, W) and Time (T) simultaneously:
         W in R^{C_out x C_in x K_t x K_h x K_w}
     This captures joint spatiotemporal patterns (e.g., optical flow, trajectory, deformation)
     hierarchically from early layers to deep layers.

2. The Three Architectural Paradigms Compared:
   - Full 3D CNN (R3D-18):
       Uses standard 3x3x3 3D convolution throughout all residual blocks.
       Parameters per kernel: 3 * 3 * 3 = 27 (times Cin * Cout).
   - Factorized (2+1)D CNN (R(2+1)D-18):
       Decomposes 3D convolution into:
         (1) Spatial 2D convolution: (1 x 3 x 3)
         (2) Temporal 1D convolution: (3 x 1 x 1)
       Parameters per kernel: (3 * 3 * 1) + (1 * 1 * 3) = 9 + 3 = 12 (approx. 55% reduction!).
       Crucial benefit: Inserts an additional non-linear activation (ReLU + BatchNorm)
       between spatial and temporal filters, doubling non-linear capacity!
   - Mixed Convolution (MC3-18):
       Applies 3D convolutions in early layers (to model fine-grained temporal motion)
       and transitions to standard 2D convolutions in deep layers (to capture semantic appearance).

3. Tensor Dimensions & Layout Convention:
   - Torchvision 3D CNNs expect: [Batch, Channels, Time, Height, Width] -> [B, C, T, H, W]
     (Note contrast with Hugging Face VideoMAE which expects [B, T, C, H, W]).
   - Standard Kinetics-400 resolution: 112 x 112 with 16 frames.

USAGE:
  # Run R(2+1)D on Apple Silicon MPS:
  python 05_spatiotemporal_3d_cnns.py --model r2plus1d --device mps

  # Run head-to-head architectural benchmark across all 3 variants:
  python 05_spatiotemporal_3d_cnns.py --model all --device mps

  # Print model layer architecture breakdown:
  python 05_spatiotemporal_3d_cnns.py --model r2plus1d --print-model
========================================================================================
"""
import argparse
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torchvision.models.video as video_models

from utils import get_device, print_device_info, read_video_pyav, resolve_data_path


# Registry of supported 3D Video Architectures in Torchvision
MODEL_REGISTRY = {
    "r3d": {
        "name": "R3D-18 (Full 3D ResNet)",
        "builder": video_models.r3d_18,
        "weights": video_models.R3D_18_Weights.DEFAULT,
        "kernel_desc": "3D Spatiotemporal (3 x 3 x 3)"
    },
    "r2plus1d": {
        "name": "R(2+1)D-18 (Factorized Spatiotemporal)",
        "builder": video_models.r2plus1d_18,
        "weights": video_models.R2Plus1D_18_Weights.DEFAULT,
        "kernel_desc": "Factorized: Spatial (1x3x3) + Temporal (3x1x1)"
    },
    "mc3": {
        "name": "MC3-18 (Mixed 3D/2D Convolution)",
        "builder": video_models.mc3_18,
        "weights": video_models.MC3_18_Weights.DEFAULT,
        "kernel_desc": "Mixed: 3D early layers, 2D deep layers"
    }
}


def load_3d_model(model_key: str, device: torch.device):
    """
    Instantiates the 3D CNN model with pretrained Kinetics-400 weights and moves to device.
    """
    cfg = MODEL_REGISTRY[model_key]
    weights = cfg["weights"]
    model = cfg["builder"](weights=weights)
    model = model.to(device)
    model.eval()
    categories = weights.meta["categories"]
    transforms = weights.transforms()
    return model, categories, transforms, cfg["name"]


def preprocess_video_frames(frames_np, transforms, device: torch.device) -> torch.Tensor:
    """
    Prepares raw video frames (T, H, W, C uint8) for torchvision 3D video models:
      1. Converts to uint8 tensor shaped (T, C, H, W).
      2. Applies official torchvision VideoClassification transform:
         - Resizes to (128, 171)
         - Center crops to (112, 112)
         - Converts to float and applies Kinetics-400 mean/std standardization
         - Permutes layout to (C, T, H, W)
      3. Unsqueezes batch dimension -> [1, C, T, H, W].
    """
    # Convert numpy array (T, H, W, C) to PyTorch tensor (T, C, H, W)
    t_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)
    
    # VideoClassification transforms expect (T, C, H, W) uint8, and return (C, T, H, W) float
    processed = transforms(t_tensor)
    
    # Add batch dimension -> [B, C, T, H, W]
    batched = processed.unsqueeze(0).to(device)
    return batched


def benchmark_model_inference(
    model: nn.Module, 
    video_tensor: torch.Tensor, 
    categories: List[str],
    repetitions: int = 5
) -> Dict:
    """
    Measures forward-pass latency, parameter count, and top-5 predictions.
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    # Warmup pass
    with torch.inference_mode():
        _ = model(video_tensor)
        if video_tensor.device.type == "mps":
            torch.mps.synchronize()
        elif video_tensor.device.type == "cuda":
            torch.cuda.synchronize()

    # Latency timing
    timings = []
    with torch.inference_mode():
        for _ in range(repetitions):
            start = time.perf_counter()
            logits = model(video_tensor)
            if video_tensor.device.type == "mps":
                torch.mps.synchronize()
            elif video_tensor.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000.0
            timings.append(elapsed)

    avg_latency = sum(timings) / len(timings)
    probs = torch.softmax(logits[0], dim=-1)
    top5_probs, top5_indices = torch.topk(probs, 5)

    top5 = []
    for prob, idx in zip(top5_probs.tolist(), top5_indices.tolist()):
        top5.append({
            "label": categories[idx],
            "probability": prob
        })

    return {
        "params": total_params,
        "latency_ms": avg_latency,
        "top5": top5
    }


def print_architecture_comparison():
    """
    Prints theoretical and structural comparison across R3D, R(2+1)D, and MC3.
    """
    print("\n" + "=" * 80)
    print(" 3D CNN ARCHITECTURAL PARADIGM COMPARISON (Kinetics-400)")
    print("=" * 80)
    print(f"{'Model':<12} | {'Paradigm':<22} | {'Kernel Structure':<30} | {'Non-Linearities'}")
    print("-" * 80)
    print(f"{'R3D-18':<12} | Full 3D Conv           | 3 x 3 x 3 (Volumetric)         | 1x per residual layer")
    print(f"{'R(2+1)D-18':<12} | Factorized Spatiotemp. | (1 x 3 x 3) + (3 x 1 x 1)      | 2x (ReLU between S and T)")
    print(f"{'MC3-18':<12} | Mixed 3D/2D Conv       | 3D early (L1-2), 2D deep (L3-4)| Hybrid")
    print("=" * 80)
    print("Key Insight: R(2+1)D has fewer FLOPs than R3D while doubling non-linear layers,")
    print("which makes it significantly easier to optimize and achieves higher accuracy.\n")


def print_layer_summary(model: nn.Module, model_key: str):
    """
    Prints key architectural blocks showing differences in conv layer definitions.
    """
    print("\n" + "-" * 80)
    print(f" LAYER ARCHITECTURE INSPECTION: {model_key.upper()}")
    print("-" * 80)
    stem = model.stem
    print(f"[*] Input Stem:\n    {stem}")
    print(f"[*] First Residual Block (layer1[0].conv1):\n    {model.layer1[0].conv1}")
    print(f"[*] Deep Residual Block (layer4[0].conv1):\n    {model.layer4[0].conv1}")
    print(f"[*] Classification Head:\n    {model.fc}")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="IF5152: 3D CNN Spatiotemporal Action Recognition Lab"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/person_bicycle.mp4",
        help="Path to input video file (defaults to data/person_bicycle.mp4)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["r3d", "r2plus1d", "mc3", "all"],
        default="r2plus1d",
        help="3D CNN architecture to run ('r3d', 'r2plus1d', 'mc3', or 'all' for comparative benchmark)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of uniformly sampled frames (default: 16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: 'mps', 'cuda', 'cpu', or 'auto'"
    )
    parser.add_argument(
        "--print-model",
        action="store_true",
        help="Print model layer architecture breakdown"
    )
    args = parser.parse_args()

    # 1. Resolve Device & Video Path
    device = get_device(args.device)
    print_device_info(device)
    print_architecture_comparison()

    video_path = resolve_data_path(args.video)
    print(f"[*] Input Video: {video_path}")
    print(f"[*] Sampling: {args.num_frames} frames via PyAV fast keyframe seek")

    # 2. Fast Keyframe Video Decoding
    frames_np = read_video_pyav(video_path, num_frames=args.num_frames)
    print(f"[*] Decoded raw video array: {frames_np.shape} (dtype: {frames_np.dtype})")

    # 3. Model Execution & Evaluation
    models_to_run = ["r3d", "r2plus1d", "mc3"] if args.model == "all" else [args.model]
    results = {}

    for m_key in models_to_run:
        print("\n" + "=" * 80)
        print(f" EXECUTING: {MODEL_REGISTRY[m_key]['name']}")
        print("=" * 80)

        model, categories, transforms, model_name = load_3d_model(m_key, device)
        
        if args.print_model:
            print_layer_summary(model, m_key)

        # Preprocess video tensor to [B=1, C=3, T=16, H=112, W=112]
        video_tensor = preprocess_video_frames(frames_np, transforms, device)
        print(f"[*] Preprocessed 5D Video Tensor Shape: {tuple(video_tensor.shape)}")
        print(f"    Layout: [Batch, Channels, Time, Height, Width]")

        # Benchmark forward pass and get Top-5
        bench = benchmark_model_inference(model, video_tensor, categories)
        results[m_key] = bench

        print(f"\n[*] Model Parameters: {bench['params']:,}")
        print(f"[*] Forward Latency on {device.type.upper()}: {bench['latency_ms']:.2f} ms")
        print("\n[*] Top-5 Kinetics-400 Action Predictions:")
        for rank, item in enumerate(bench["top5"], start=1):
            bar = "#" * int(item["probability"] * 30)
            print(f"    {rank}. {item['label']:<24} {item['probability']*100:5.2f}%  [{bar:<30}]")

    # 4. Comparative Summary Table (when running multiple or all)
    if len(models_to_run) > 1:
        print("\n" + "=" * 85)
        print(" HEAD-TO-HEAD BENCHMARK SUMMARY")
        print("=" * 85)
        print(f"{'Model':<12} | {'Params':<12} | {'Latency (ms)':<14} | {'Top-1 Prediction':<24} | {'Prob (%)'}")
        print("-" * 85)
        for m_key in models_to_run:
            b = results[m_key]
            top1 = b["top5"][0]
            print(
                f"{m_key.upper():<12} | {b['params']:<12,} | {b['latency_ms']:<14.2f} | "
                f"{top1['label']:<24} | {top1['probability']*100:.2f}%"
            )
        print("=" * 85)
        print("Key Pedagogical Takeaway:")
        print("Notice how R(2+1)D provides competitive or superior accuracy with structured")
        print("spatial-temporal factorization, while maintaining high efficiency on hardware.")


if __name__ == "__main__":
    main()
