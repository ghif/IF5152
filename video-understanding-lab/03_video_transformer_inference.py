"""
========================================================================================
IF5152: Video Understanding - Session 3 Lab
File: 03_video_transformer_inference.py
========================================================================================

PEDAGOGICAL & THEORETICAL FOUNDATIONS:
----------------------------------------------------------------------------------------
1. Self-Attention in Spatio-Temporal Space:
   - Convolutions have a local receptive field. To capture interactions between distant
     regions in space and time, early methods used Non-Local Blocks.

2. Vision Transformers for Video (TimeSformer / VideoMAE):
   - ViT (Vision Transformer) applies self-attention over flattened 2D spatial patches.
   - For Video: A video clip of T frames is split into 3D "tubelet" patches (e.g., 2x16x16).
   - Problem: Full space-time self-attention across all tokens has quadratic complexity O(N^2),
     which is computationally prohibitive.
   - Solution: "Divided Space-Time Attention" (TimeSformer) or Factorized Attention:
     First attend within each frame across SPACE, then attend across TIME for each patch!

3. Industrial Perspective: Foundation Models vs Training from Scratch:
   - Training 3D CNNs or Video Transformers from scratch requires weeks of cluster compute.
   - In industry, teams fine-tune or run inference on pretrained Video Foundation Models
     (e.g., VideoMAE, TimeSformer, or MViTv2) pretrained on datasets like Kinetics-400.

USAGE:
  python 03_video_transformer_inference.py --video data/person_bicycle.mp4
  python 03_video_transformer_inference.py --video data/person_bicycle.mp4 --device mps
========================================================================================
"""
import argparse
import time
import numpy as np
import torch
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

from utils import get_device, print_device_info, read_video_pyav, resolve_data_path


def main():
    parser = argparse.ArgumentParser(
        description="IF5152 Lecture 11 Lab: Pretrained Video Transformer (VideoMAE) Inference"
    )
    parser.add_argument("--video", type=str, default="data/person_bicycle.mp4",
                        help="Path to input video clip (MP4 format)")
    parser.add_argument("--model", type=str, default="MCG-NJU/videomae-base-finetuned-kinetics",
                        help="Hugging Face checkpoint (Pretrained on Kinetics-400)")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of uniformly sampled frames (VideoMAE standard is 16)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Target device: 'auto' | 'mps' | 'cuda' | 'cpu'")
    args = parser.parse_args()

    # 1. Hardware Inspection & Video Path Resolution
    device = get_device(args.device)
    print_device_info(device)
    video_path = resolve_data_path(args.video)

    # 2. Load the Preprocessing Pipeline
    # The processor handles video normalization, spatial resizing to 224x224, and tensor conversion
    print(f"\n--- [STEP 1] Loading Preprocessor for '{args.model}' ---")
    processor = AutoImageProcessor.from_pretrained(args.model)

    # 3. Decode Video Frames using PyAV (Fast Seek)
    print(f"\n--- [STEP 2] Decoding Video Clip ---")
    print(f"Sampling {args.num_frames} frames from '{video_path}'...")
    raw_frames = read_video_pyav(video_path, num_frames=args.num_frames, fast_seek=True)
    print(f"Raw decoded array shape: {raw_frames.shape} (Frames, Height, Width, RGB Channels)")

    # 4. Prepare Inputs for the Video Transformer
    # VideoMAE expects a list of RGB numpy arrays (or PIL images) of shape [T, H, W, C]
    inputs = processor(list(raw_frames), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Expected tensor shape: [Batch=1, Num_Frames=16, Channels=3, Height=224, Width=224]
    print(f"Processed Transformer Input Tensor: {tuple(inputs['pixel_values'].shape)}")

    # 5. Load the Pretrained Video Transformer Model
    print(f"\n--- [STEP 3] Loading Pretrained VideoMAE Model onto {device.type.upper()} ---")
    model = VideoMAEForVideoClassification.from_pretrained(args.model).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,} (Pretrained on 400 Kinetics Action Classes)")

    # 6. Forward Pass & Attention Evaluation
    print("\n--- [STEP 4] Running Spatio-Temporal Attention Forward Pass ---")
    with torch.no_grad():
        t0 = time.perf_counter()
        outputs = model(**inputs)

        # Synchronize hardware timer
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0

        logits = outputs.logits                     # Shape: [1, 400]
        probs = torch.softmax(logits, dim=-1)       # Convert raw logits to probabilities

    print(f"Forward Pass Latency: {infer_ms:.2f} ms")

    # 7. Display Top-5 Ranked Action Predictions
    top5_probs, top5_indices = torch.topk(probs[0], k=5)

    print("\n==================================================")
    print(" TOP-5 KINETICS-400 ACTION PREDICTIONS")
    print("==================================================")
    for rank, (p, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
        label_name = model.config.id2label.get(idx.item(), f"class_{idx.item()}")
        bar = "█" * int(p.item() * 40)
        print(f"  #{rank}  {label_name:<30} : {p.item() * 100:5.2f}%  {bar}")
    print("==================================================")
    print("\n[SUCCESS] Session 3 (Video Transformer) demo completed.")


if __name__ == "__main__":
    main()
