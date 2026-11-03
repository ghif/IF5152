"""
========================================================================================
IF5152: Video Understanding - Session 2 Lab (Part B)
File: 03_zeroshot_video_search.py
========================================================================================

PEDAGOGICAL & THEORETICAL FOUNDATIONS:
----------------------------------------------------------------------------------------
1. Lecture Slides 99-105: "Multimodal Video Understanding"
   - Earlier video architectures were "closed-set": trained to recognize a fixed list of
     classes (e.g., 101 UCF classes or 400 Kinetics classes).
   - Limitation: Real-world video applications (content moderation, CCTV monitoring,
     video search engines like YouTube/TikTok) require recognizing actions that were
     NEVER seen during training.

2. Dual-Encoder Vision-Language Matching (CLIP):
   - A Vision Encoder maps image/video frames into a shared D-dimensional embedding space.
   - A Text Encoder maps arbitrary natural language descriptions into the same space.
   - Zero-Shot Video Classification:
       (a) Sample T frames across the video duration.
       (b) Extract frame embeddings [T, D] using the Vision Encoder.
       (c) Temporal Aggregation: Average-pool frame embeddings to produce a single
           compact Video Representation [1, D].
       (d) Compute Cosine Similarity between the video vector and candidate text queries.
       (e) Apply temperature-scaled Softmax to produce matching probabilities.

3. Industrial Perspective:
   - This exact approach powers modern large-scale video retrieval and semantic search,
     requiring ZERO training labels for new domains or new action queries!

USAGE:
  python 03_zeroshot_video_search.py --video data/person_bicycle.mp4
  python 03_zeroshot_video_search.py --video data/person_bicycle.mp4 --device mps
========================================================================================
"""
import argparse
import time
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

from utils import get_device, print_device_info, read_video_pyav, resolve_data_path


# Candidate open-vocabulary queries for demonstration.
# Note: You can add any arbitrary action prompt here to show students zero-shot adaptability!
DEFAULT_QUERIES = [
    "a person riding a bicycle outdoors",
    "a person playing an acoustic guitar",
    "a chef preparing food in a kitchen",
    "cars driving on a highway road",
    "a person applying facial makeup",
    "two people having a conversation"
]


def main():
    parser = argparse.ArgumentParser(
        description="IF5152 Lecture 11 Lab: Zero-Shot Video Understanding with Multimodal Foundation Models (CLIP)"
    )
    parser.add_argument("--video", type=str, default="data/person_bicycle.mp4",
                        help="Path to input video clip (MP4 format)")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32",
                        help="Pretrained CLIP vision-language checkpoint")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Number of uniformly sampled frames across video duration")
    parser.add_argument("--device", type=str, default="auto",
                        help="Target device: 'auto' | 'mps' | 'cuda' | 'cpu'")
    args = parser.parse_args()

    # 1. Hardware Inspection & Video Path Resolution
    device = get_device(args.device)
    print_device_info(device)
    video_path = resolve_data_path(args.video)

    # 2. Load the Pretrained Vision-Language Foundation Model
    print(f"\n--- [STEP 1] Loading Multimodal Foundation Model: {args.model} ---")
    model = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Vision + Text Encoder Parameters: {total_params:,}")

    # 3. Decode Sampled Frames from Video
    print(f"\n--- [STEP 2] Decoding {args.num_frames} Video Frames ---")
    raw_frames = read_video_pyav(video_path, num_frames=args.num_frames, fast_seek=True)
    # Convert numpy arrays to PIL Images for the CLIP processor
    pil_frames = [Image.fromarray(f) for f in raw_frames]
    print(f"Decoded {len(pil_frames)} frames across the video duration.")

    # 4. Define Candidate Open-Vocabulary Text Queries
    print("\n--- [STEP 3] Defining Candidate Open-Vocabulary Queries ---")
    for i, q in enumerate(DEFAULT_QUERIES, start=1):
        print(f"  Query #{i}: \"{q}\"")

    # 5. Tokenize Text and Preprocess Frames
    print("\n--- [STEP 4] Encoding Visual and Textual Modalities ---")
    inputs = processor(
        text=DEFAULT_QUERIES,
        images=pil_frames,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        t0 = time.perf_counter()

        # Extract Visual Features:
        # Note: transformers v5+ returns BaseModelOutputWithPooling; extract pooler_output
        img_out = model.get_image_features(inputs["pixel_values"])
        image_features = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out  # Shape: [T, 512]

        # Extract Text Features:
        txt_out = model.get_text_features(inputs["input_ids"])
        text_features = txt_out.pooler_output if hasattr(txt_out, "pooler_output") else txt_out    # Shape: [Num_Queries, 512]

        # ----------------------------------------------------------------------
        # Mathematical Step: L2 Normalization onto the Unit Hypersphere
        # Formula: v_norm = v / ||v||_2
        # ----------------------------------------------------------------------
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # ----------------------------------------------------------------------
        # Temporal Aggregation (Late Fusion on Embeddings):
        # We average the L2-normalized frame vectors across time to obtain a single
        # compact representation for the entire video clip:
        #   v_video = Mean(v_frame_1, v_frame_2, ..., v_frame_T)
        # ----------------------------------------------------------------------
        video_feature = image_features.mean(dim=0, keepdim=True)  # Shape: [1, 512]
        video_feature = video_feature / video_feature.norm(dim=-1, keepdim=True)

        # ----------------------------------------------------------------------
        # Cosine Similarity & Softmax Probability:
        # Cosine Similarity between video and queries = video_feature @ text_features.T
        # Temperature scaling (100.0) sharpens the softmax distribution.
        # ----------------------------------------------------------------------
        similarity = (video_feature @ text_features.T).squeeze(0)  # Shape: [Num_Queries]
        probs = torch.softmax(similarity * 100.0, dim=-1)

        # Synchronize hardware timer
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # 6. Display Ranked Zero-Shot Results
    print(f"\n--- [STEP 5] Zero-Shot Retrieval Results (Computed in {elapsed_ms:.2f} ms) ---")
    print("=" * 68)
    ranked = sorted(zip(DEFAULT_QUERIES, probs.cpu().numpy()), key=lambda x: x[1], reverse=True)
    for rank, (query, score) in enumerate(ranked, start=1):
        bar = "█" * int(score * 30)
        print(f"  #{rank}  {query:<45} : {score * 100:6.2f}%  {bar}")
    print("=" * 68)
    print("\n[SUCCESS] Session 2 (Zero-Shot Video Search) demo completed.")


if __name__ == "__main__":
    main()
