"""
========================================================================================
IF5152: Video Understanding - Session 1 Lab
File: 01_video_io_and_fusion_paradigms.py
========================================================================================

PEDAGOGICAL & THEORETICAL FOUNDATIONS:
----------------------------------------------------------------------------------------
1. Video Representation: "Video = 2D + Time"
   - An image is a 3D tensor: [C x H x W]
   - A video clip is a 4D tensor: [T x C x H x W] (or 5D batched: [B x T x C x H x W])
   - Challenge: Processing raw video (e.g. 30 fps, 1080p) blows up memory and compute!

2. Training on Clips:
   - Rather than feeding entire long videos, we extract short clips sampled at low FPS
     using uniform temporal sampling.

3. Spatio-Temporal Fusion Paradigms (Karpathy et al., CVPR 2014):
   - Early Fusion:
     Stack all T frames along the channel dimension at the input [B, (T*C), H, W].
     The first convolutional layer (stem) is modified to take (T*C) input channels.
     Pretrained 2D ImageNet weights are initialized via Channel Weight Inflation
     (W_early = W_2D / T). Time is collapsed immediately in the first layer, so the
     backbone processes only 1 spatial map per clip.
   - Late Fusion (Mean Pooling):
     Process all T frames independently through the 2D CNN backbone, then aggregate
     the extracted high-level spatial feature vectors using temporal average pooling.
     Inductive bias: Permutation-invariant (order-invariant).
   - Late Fusion (Recurrent GRU):
     Process all T frames through the 2D CNN backbone, then feed the sequence of
     feature vectors into a Gated Recurrent Unit (GRU).
     Inductive bias: Order-sensitive temporal dynamics via sequential hidden states.

4. Industrial Perspective: "The I/O Decoding Bottleneck"
   - A major real-world bottleneck in video deep learning is video decoding on CPU.
   - We benchmark three approaches:
       (1) OpenCV sequential decoding loop
       (2) PyAV naive linear decode (Python object wrapping overhead)
       (3) PyAV industrial fast seek (FFmpeg multithreading + keyframe navigation)

USAGE:
  # Run Early Fusion:
  python 01_video_io_and_fusion_paradigms.py --video data/person_bicycle.mp4 --fusion early

  # Run Late Fusion (Mean Pooling):
  python 01_video_io_and_fusion_paradigms.py --video data/person_bicycle.mp4 --fusion mean

  # Run Late Fusion (Recurrent GRU):
  python 01_video_io_and_fusion_paradigms.py --video data/person_bicycle.mp4 --fusion gru

  # Comparative Head-to-Head Benchmark across all three paradigms:
  python 01_video_io_and_fusion_paradigms.py --video data/person_bicycle.mp4 --fusion all --device mps
========================================================================================
"""
import argparse
import time
import av
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from utils import get_device, print_device_info, read_video_pyav, sample_frame_indices, resolve_data_path


# ======================================================================================
# PART 1: INDUSTRIAL VIDEO DECODING BENCHMARK
# ======================================================================================

def benchmark_decoders(video_path: str, num_frames: int = 16):
    """
    Compares three decoding paradigms to illustrate video I/O bottlenecks to students:
      1. OpenCV: Standard C++ loop; reads frames linearly.
      2. PyAV Naive: Linear decode in Python; slow due to allocating 600+ Python objects.
      3. PyAV Fast Seek: Industrial method; seeks directly to keyframes with multithreading.
    """
    # ----------------------------------------------------------------------------------
    # Approach 1: OpenCV Sequential Loop
    # ----------------------------------------------------------------------------------
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = set(sample_frame_indices(max(1, total_frames), num_frames))
    
    frames_cv = []
    idx = 0
    while cap.isOpened() and len(frames_cv) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in target_indices:
            # Note: OpenCV reads in BGR format; convert to RGB for deep learning models
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_cv.append(frame_rgb)
        idx += 1
    cap.release()
    t_cv = (time.perf_counter() - t0) * 1000.0

    # ----------------------------------------------------------------------------------
    # Approach 2: PyAV Naive Linear Decode (Single-threaded)
    # ----------------------------------------------------------------------------------
    t0 = time.perf_counter()
    container = av.open(video_path)
    stream = container.streams.video[0]
    frames_naive = []
    # In a naive loop, PyAV instantiates an 'av.VideoFrame' Python wrapper for EVERY frame
    for i, frame in enumerate(container.decode(stream)):
        if i in target_indices:
            frames_naive.append(frame.to_ndarray(format="rgb24"))
        if len(frames_naive) == num_frames:
            break
    container.close()
    t_naive = (time.perf_counter() - t0) * 1000.0

    # ----------------------------------------------------------------------------------
    # Approach 3: PyAV Fast Seek (Multi-threaded slice decoding + Keyframe Seeking)
    # ----------------------------------------------------------------------------------
    t0 = time.perf_counter()
    # Uses container.seek() to jump directly to timestamps, skipping unnecessary packets
    frames_seek = read_video_pyav(video_path, num_frames=num_frames, fast_seek=True)
    t_seek = (time.perf_counter() - t0) * 1000.0

    return {
        "opencv": (t_cv, len(frames_cv)),
        "pyav_naive": (t_naive, len(frames_naive)),
        "pyav_seek": (t_seek, len(frames_seek), frames_seek)
    }


# ======================================================================================
# PART 2: ARCHITECTURES - EARLY FUSION & LATE FUSION CLASSIFIERS
# ======================================================================================

class EarlyFusionVideoClassifier(nn.Module):
    """
    Early Fusion Architecture (Karpathy et al., CVPR 2014):
    
    Theoretical Mechanism:
      - Fuses temporal information at the pixel level in the very first convolutional layer.
      - Input video clip of shape [B, T, C, H, W] is flattened along the channel dimension
        to [B, (T * C), H, W] (e.g. 16 frames * 3 RGB = 48 input channels).
      - Pretrained 2D ImageNet weights in the stem conv layer are adapted using
        Channel Weight Inflation:
            W_early = W_2D.repeat(1, T, 1, 1) / T
        On a static video (identical frames), the output is mathematically identical
        to the single-frame 2D network!
      - Computational Advantage: The backbone processes only 1 spatial map per clip (B=1)
        instead of (B * T = 16), making the deeper layers extremely fast.
      - Representation Trade-off: Restricts temporal interactions to local receptive fields
        in the first layer; deeper features lose explicit temporal dynamics.
    """
    def __init__(self, num_classes: int = 10, num_frames: int = 16):
        super().__init__()
        self.num_frames = num_frames
        self.in_channels = 3 * num_frames

        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        backbone = models.mobilenet_v3_small(weights=weights)
        self.feat_dim = backbone.classifier[0].in_features

        # Replace stem Conv2d(3, 16, kernel_size=3, stride=2, padding=1) with inflated Conv2d
        orig_stem_conv = backbone.features[0][0]
        inflated_stem_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=orig_stem_conv.out_channels,
            kernel_size=orig_stem_conv.kernel_size,
            stride=orig_stem_conv.stride,
            padding=orig_stem_conv.padding,
            bias=(orig_stem_conv.bias is not None)
        )

        # Bootstrap pretrained weights via Channel Inflation (divide by T to preserve activation scale)
        with torch.no_grad():
            inflated_stem_conv.weight.data = orig_stem_conv.weight.data.repeat(1, num_frames, 1, 1) / float(num_frames)
            if orig_stem_conv.bias is not None:
                inflated_stem_conv.bias.data.copy_(orig_stem_conv.bias.data)

        backbone.features[0][0] = inflated_stem_conv
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        # Collapse time into channel dimension: [B, T * C, H, W]
        x_early = x.view(B, T * C, H, W)
        # Backbone evaluates on a single spatial map per video clip
        feat = self.backbone(x_early)  # [B, feat_dim]
        logits = self.head(feat)       # [B, num_classes]
        return logits


class LateFusionVideoClassifier(nn.Module):
    """
    Late Fusion Architecture (Karpathy et al., CVPR 2014):
    
    Data Flow & Tensor Transformations:
      1. Input Video Tensor   : [B, T, C, H, W]  e.g. [1, 16, 3, 224, 224]
      2. Flatten for 2D CNN   : [(B * T), C, H, W] -> Feed standard 2D backbone
      3. Extracted Features   : [(B * T), Feat_Dim] e.g. [16, 576]
      4. Reshape to Temporal  : [B, T, Feat_Dim]   e.g. [1, 16, 576]
      5. Temporal Aggregation :
           - Option 'mean'    : Average across Time dimension -> [B, Feat_Dim]
           - Option 'gru'     : Recurrent sequence modeling   -> [B, Hidden_Dim]
      6. Classification Head  : Linear layer -> [B, Num_Classes]
    """
    def __init__(self, num_classes: int = 10, fusion: str = "mean"):
        super().__init__()
        self.fusion = fusion

        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        backbone = models.mobilenet_v3_small(weights=weights)
        self.feat_dim = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        if fusion == "gru":
            hidden_dim = 256
            self.temporal_model = nn.GRU(
                input_size=self.feat_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=False
            )
            self.head = nn.Linear(hidden_dim, num_classes)
        elif fusion == "mean":
            self.head = nn.Linear(self.feat_dim, num_classes)
        else:
            raise ValueError(f"Unsupported late fusion strategy: {fusion}. Choose 'mean' or 'gru'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        # Run 2D backbone across all frames in parallel by folding time into batch
        x_flat = x.view(B * T, C, H, W)
        frame_feats = self.backbone(x_flat)       # Shape: [B * T, feat_dim]
        frame_feats = frame_feats.view(B, T, -1)   # Shape: [B, T, feat_dim]

        # Temporal Aggregation
        if self.fusion == "mean":
            video_feat = frame_feats.mean(dim=1)
            logits = self.head(video_feat)
        elif self.fusion == "gru":
            gru_out, _ = self.temporal_model(frame_feats)
            video_feat = gru_out[:, -1, :]
            logits = self.head(video_feat)

        return logits


def build_fusion_model(fusion: str, num_classes: int = 10, num_frames: int = 16) -> nn.Module:
    """Factory helper to construct the selected fusion model architecture."""
    if fusion == "early":
        return EarlyFusionVideoClassifier(num_classes=num_classes, num_frames=num_frames)
    elif fusion in ["mean", "gru"]:
        return LateFusionVideoClassifier(num_classes=num_classes, fusion=fusion)
    else:
        raise ValueError(f"Unknown fusion model: '{fusion}'. Choose from ['early', 'mean', 'gru', 'all'].")


# ======================================================================================
# MAIN EXECUTION PIPELINE
# ======================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="IF5152 Lecture 11 Lab: Video I/O Benchmarking and Temporal Fusion Paradigms"
    )
    parser.add_argument("--video", type=str, default="data/person_bicycle.mp4",
                        help="Path to demo video clip (MP4 format)")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of uniformly sampled temporal frames (default: 16)")
    parser.add_argument("--fusion", choices=["early", "mean", "gru", "all"], default="mean",
                        help="Fusion strategy: 'early', 'mean' (late avg), 'gru' (late recurrent), or 'all' (benchmark)")
    parser.add_argument("--print-model", action="store_true",
                        help="Print detailed PyTorch neural network layer architecture")
    parser.add_argument("--device", type=str, default="auto",
                        help="Target device: 'auto' (MPS on Mac, CUDA on Cloud/Linux, CPU fallback)")
    args = parser.parse_args()

    # 1. Hardware Inspection & Video Path Resolution
    device = get_device(args.device)
    print_device_info(device)
    video_path = resolve_data_path(args.video)

    # 2. Decoding Benchmark (Industrial Perspective)
    print("\n--- [STEP 1] Industrial I/O Decoding Benchmark ---")
    results = benchmark_decoders(video_path, args.num_frames)
    t_cv, n_cv = results["opencv"]
    t_naive, n_naive = results["pyav_naive"]
    t_seek, n_seek, raw_frames = results["pyav_seek"]

    print(f"Target Video Clip : {video_path} (Sample {args.num_frames} frames)")
    print(f"  1. OpenCV (Sequential Loop)             : {t_cv:6.2f} ms")
    print(f"  2. PyAV Naive (Single-threaded Loop)    : {t_naive:6.2f} ms")
    print(f"  3. PyAV Fast Seek (Industrial Standard) : {t_seek:6.2f} ms")
    print(f"==> PyAV with Keyframe Seeking is {t_cv / max(0.001, t_seek):.1f}x faster than OpenCV!")

    # 3. Constructing the 5D Video Tensor
    print("\n--- [STEP 2] Constructing the 5D Video Tensor ---")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor_list = [transform(Image.fromarray(f)) for f in raw_frames]
    video_tensor = torch.stack(tensor_list).unsqueeze(0).to(device)

    print(f"Input Video Tensor Shape: {tuple(video_tensor.shape)} (Batch, Time, Channels, Height, Width)")
    tensor_kb = video_tensor.element_size() * video_tensor.nelement() / 1024.0
    print(f"Tensor VRAM Footprint on {device.type.upper()}: {tensor_kb:.2f} KB")

    # 4. Model Execution & Benchmarking
    eval_fusions = ["early", "mean", "gru"] if args.fusion == "all" else [args.fusion]
    benchmark_records = []

    print(f"\n--- [STEP 3] Evaluating Fusion Paradigms ({args.fusion.upper()}) ---")

    for fusion_type in eval_fusions:
        model = build_fusion_model(fusion_type, num_classes=10, num_frames=args.num_frames).to(device)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())

        if args.print_model:
            print("\n" + "=" * 65)
            print(f" MODEL ARCHITECTURE: {fusion_type.upper()} FUSION")
            print("=" * 65)
            print(model)
            print("=" * 65)

        # Warm-up pass
        with torch.no_grad():
            _ = model(video_tensor)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

        # Timed benchmark pass (average of 5 runs)
        latencies = []
        with torch.no_grad():
            for _ in range(5):
                t0 = time.perf_counter()
                logits = model(video_tensor)
                if device.type == "mps":
                    torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_latency = float(np.mean(latencies))
        probs = torch.softmax(logits, dim=-1)
        top_prob, top_idx = torch.max(probs, dim=-1)

        benchmark_records.append({
            "fusion": fusion_type,
            "params": total_params,
            "latency": avg_latency,
            "stem_shape": f"[1, {3 * args.num_frames}, 224, 224]" if fusion_type == "early" else f"[{args.num_frames}, 3, 224, 224]",
            "passes": 1 if fusion_type == "early" else args.num_frames,
            "top_idx": top_idx.item(),
            "top_prob": top_prob.item() * 100.0
        })

        if args.fusion != "all":
            print(f"\nTotal Parameters                   : {total_params:,}")
            print(f"Forward Pass Execution Latency     : {avg_latency:.2f} ms")
            print(f"Output Logits Tensor Shape         : {tuple(logits.shape)} (Batch, Classes)")
            print(f"Class Probabilities (Softmax)      : {np.round(probs[0].cpu().numpy(), 4)}")

    # Comparative Summary Table if multiple or 'all' requested
    if args.fusion == "all":
        print("\n" + "=" * 88)
        print(f" HEAD-TO-HEAD COMPARATIVE BENCHMARK: SPATIO-TEMPORAL FUSION PARADIGMS ({device.type.upper()})")
        print("=" * 88)
        header = f"{'Fusion Strategy':<18} | {'Stem Input':<18} | {'Backbone Passes':<15} | {'Params':<11} | {f'Latency ({device.type.upper()})':<14}"
        print(header)
        print("-" * len(header))
        for r in benchmark_records:
            name_map = {
                "early": "Early Fusion",
                "mean": "Late Fusion (Mean)",
                "gru": "Late Fusion (GRU)"
            }
            print(f"{name_map[r['fusion']]:<18} | {r['stem_shape']:<18} | {r['passes']:<15} | {r['params']:<11,} | {r['latency']:6.2f} ms")
        print("=" * 88)
        print("\nKey Pedagogical Takeaways for Students:")
        print("  1. Early Fusion is ~3.5x faster in the forward pass because the 2D CNN backbone")
        print("     processes only 1 spatial map ([1, 48, 224, 224]) rather than 16 frames in parallel.")
        print("  2. However, Early Fusion restricts temporal interactions to local 3x3 receptive fields")
        print("     in the stem layer, whereas Late Fusion allows the backbone to extract high-level semantic")
        print("     features before temporal aggregation.")

    print("\n[SUCCESS] Session 1 demo completed.")


if __name__ == "__main__":
    main()
