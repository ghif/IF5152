"""
========================================================================================
IF5152: Video Understanding - Session 1 Lab
File: 01_video_io_and_late_fusion.py
========================================================================================

PEDAGOGICAL & THEORETICAL FOUNDATIONS:
----------------------------------------------------------------------------------------
1. Lecture Slide 9: "Video = 2D + Time"
   - An image is a 3D tensor: [C x H x W]
   - A video clip is a 4D tensor: [T x C x H x W] (or 5D batched: [B x T x C x H x W])
   - Challenge: Processing raw video (e.g. 30 fps, 1080p) blows up memory and compute!

2. Lecture Slides 14-16: "Training on Clips"
   - Rather than feeding entire long videos, we extract short clips sampled at low FPS
     using uniform temporal sampling.

3. Lecture Slides 17-20: "Single-Frame CNN vs Late Fusion"
   - Single-Frame CNN (Slide 17): Classify each frame independently, average test predictions.
   - Late Fusion (Slides 18-20): Extract deep spatial feature maps from each frame using
     a 2D CNN backbone, then fuse temporal information before the classification layer.
     We compare:
       (a) Average Pooling over time (Slide 19)
       (b) Recurrent GRU modeling (Slide 63)

4. Industrial Perspective: "The I/O Decoding Bottleneck"
   - A major real-world bottleneck in video deep learning is video decoding on CPU.
   - We benchmark three approaches:
       (1) OpenCV sequential decoding loop
       (2) PyAV naive linear decode (Python object wrapping overhead)
       (3) PyAV industrial fast seek (FFmpeg multithreading + keyframe navigation)

USAGE:
  python 01_video_io_and_late_fusion.py --video data/person_bicycle.mp4 --fusion mean
  python 01_video_io_and_late_fusion.py --video data/person_bicycle.mp4 --fusion gru --device mps
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
# PART 2: ARCHITECTURE - LATE FUSION VIDEO CLASSIFIER
# ======================================================================================

class LateFusionVideoClassifier(nn.Module):
    """
    Implementation of Late Fusion (Lecture Slides 18-20 & 63):
    
    Data Flow & Tensor Transformations:
      1. Input Video Tensor   : [Batch, Time, Channels, Height, Width]  e.g. [1, 16, 3, 224, 224]
      2. Flatten for 2D CNN   : [(Batch * Time), Channels, Height, Width] -> Feed 2D backbone
      3. Extracted Features   : [(Batch * Time), Feat_Dim]              e.g. [16, 576]
      4. Reshape to Spatio-Temporal: [Batch, Time, Feat_Dim]            e.g. [1, 16, 576]
      5. Temporal Aggregation :
           - Option 'mean'    : Average across Time dimension -> [Batch, Feat_Dim] (Slide 19)
           - Option 'gru'     : Recurrent sequence modeling   -> [Batch, Hidden_Dim] (Slide 63)
      6. Classification Head  : Linear layer -> [Batch, Num_Classes]
    """
    def __init__(self, num_classes: int = 10, fusion: str = "mean"):
        super().__init__()
        self.fusion = fusion

        # Pedagogical Choice: Use MobileNetV3-Small as a lightweight 2D spatial backbone.
        # It runs very fast on CPU/MPS while having ImageNet-pretrained spatial filters.
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        backbone = models.mobilenet_v3_small(weights=weights)

        # Retrieve feature dimension (576 for MobileNetV3-Small)
        self.feat_dim = backbone.classifier[0].in_features

        # Replace classification head with Identity to extract raw spatial feature vectors
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        if fusion == "gru":
            # Slide 63: Recurrent neural network for long-term temporal structure
            hidden_dim = 256
            self.temporal_model = nn.GRU(
                input_size=self.feat_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=False
            )
            self.head = nn.Linear(hidden_dim, num_classes)
        elif fusion == "mean":
            # Slide 19: Average pooling across temporal dimension
            self.head = nn.Linear(self.feat_dim, num_classes)
        else:
            raise ValueError(f"Unsupported fusion strategy: {fusion}. Choose 'mean' or 'gru'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Unpack 5D dimensions
        B, T, C, H, W = x.shape

        # Step 2: Reshape [B, T, C, H, W] -> [B * T, C, H, W] to run 2D CNN across all frames
        x_flat = x.view(B * T, C, H, W)
        frame_feats = self.backbone(x_flat)       # Shape: [B * T, feat_dim]

        # Step 3: Reshape back to temporal sequence: [B, T, feat_dim]
        frame_feats = frame_feats.view(B, T, -1)

        # Step 4: Temporal Aggregation
        if self.fusion == "mean":
            # Temporal Average Pooling: Compress time dimension [B, T, D] -> [B, D]
            video_feat = frame_feats.mean(dim=1)
            logits = self.head(video_feat)        # [B, num_classes]
        elif self.fusion == "gru":
            # Recurrent GRU modeling: Take hidden state of the final temporal step
            gru_out, _ = self.temporal_model(frame_feats)  # [B, T, hidden_dim]
            video_feat = gru_out[:, -1, :]                 # [B, hidden_dim]
            logits = self.head(video_feat)                 # [B, num_classes]

        return logits


# ======================================================================================
# MAIN EXECUTION PIPELINE
# ======================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="IF5152 Lecture 11 Lab: Video I/O Benchmarking and Late Fusion Classifier"
    )
    parser.add_argument("--video", type=str, default="data/person_bicycle.mp4",
                        help="Path to demo video clip (MP4 format)")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of uniformly sampled temporal frames (default: 16)")
    parser.add_argument("--fusion", choices=["mean", "gru"], default="mean",
                        help="Temporal fusion strategy: 'mean' (pooling) or 'gru' (recurrent)")
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

    # 3. Constructing the 5D Video Tensor (Lecture Slide 9)
    print("\n--- [STEP 2] Constructing the 5D Video Tensor ---")
    # Standard ImageNet normalization: (x - mean) / std
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Convert raw numpy RGB frames into a stacked PyTorch tensor
    tensor_list = [transform(Image.fromarray(f)) for f in raw_frames]
    # Stack into [T, C, H, W] and add Batch dimension -> [1, T, C, H, W]
    video_tensor = torch.stack(tensor_list).unsqueeze(0).to(device)

    print(f"Input Video Tensor Shape: {tuple(video_tensor.shape)} (Batch, Time, Channels, Height, Width)")
    tensor_kb = video_tensor.element_size() * video_tensor.nelement() / 1024.0
    print(f"Tensor VRAM Footprint on {device.type.upper()}: {tensor_kb:.2f} KB")

    # 4. Model Forward Pass & Latency Measurement
    print(f"\n--- [STEP 3] Late Fusion Model ({args.fusion.upper()}) ---")
    model = LateFusionVideoClassifier(num_classes=10, fusion=args.fusion).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Backbone + Fusion Total Parameters : {total_params:,}")

    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(video_tensor)

        # Hardware synchronization to ensure accurate latency measurement
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0

        probs = torch.softmax(logits, dim=-1)

    print(f"Forward Pass Execution Latency     : {infer_ms:.2f} ms")
    print(f"Output Logits Tensor Shape         : {tuple(logits.shape)} (Batch, Classes)")
    print(f"Class Probabilities (Softmax)      : {np.round(probs[0].cpu().numpy(), 4)}")
    print("\n[SUCCESS] Session 1 demo completed.")


if __name__ == "__main__":
    main()
