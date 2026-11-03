"""
utils.py: Shared educational utilities for video decoding and hardware acceleration.
=====================================================================================
Pedagogical Purpose:
  - Demonstrates cross-platform hardware acceleration in PyTorch (Apple Silicon MPS,
    Cloud/NVIDIA CUDA, and standard CPU fallback).
  - Demonstrates industrial video decoding techniques using FFmpeg (via PyAV) to show
    students why random frame access requires timestamp seeking rather than naive loops.
=====================================================================================
"""
import os
import av
import numpy as np
import torch


def resolve_data_path(file_path: str = "person_bicycle.mp4") -> str:
    """
    Intelligently resolves data path whether the script is invoked from the
    repository root (IF5152/) or from inside a child directory (video-understanding-lab/).
    """
    filename = os.path.basename(file_path)
    root_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    candidates = [
        file_path,                                    # Literal path passed by user
        os.path.join(root_data_dir, filename),        # <root>/data/<filename>
        os.path.join("data", filename),               # ./data/<filename>
        os.path.join("..", "data", filename),          # ../data/<filename>
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Fallback to root data path (useful for downloader target)
    return os.path.join(root_data_dir, filename)


def get_device(requested: str = "auto") -> torch.device:
    """
    Selects the optimal hardware accelerator based on available hardware.
    
    Pedagogical Note for Students:
      - 'mps'  (Metal Performance Shaders): Apple Silicon GPU (M1-M4) with Unified Memory.
      - 'cuda' : NVIDIA GPU accelerator on Linux/Windows workstations or Google Colab.
      - 'cpu'  : Universal fallback when no dedicated GPU is available.
    """
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    dev = torch.device(requested)
    # Graceful fallback validations
    if dev.type == "mps" and not torch.backends.mps.is_available():
        print("[WARNING] MPS requested but Apple Silicon GPU is not available. Falling back to CPU.")
        return torch.device("cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but NVIDIA GPU is not available. Falling back to CPU.")
        return torch.device("cpu")
    return dev


def print_device_info(device: torch.device):
    """
    Inspects and prints current PyTorch environment and accelerator specs.
    Demonstrates hardware introspection to students.
    """
    print("=" * 65)
    print(f" PyTorch Version : {torch.__version__}")
    print(f" Active Device   : {device.type.upper()}")
    if device.type == "mps":
        print(f" Backend         : Apple Silicon Metal Performance Shaders (MPS)")
        print(f" Note            : Leveraging Unified Memory Architecture (UMA)")
    elif device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f" GPU Name        : {gpu_name}")
        print(f" Dedicated VRAM  : {total_mem:.2f} GB")
    else:
        print(f" Backend         : Standard Host CPU (No hardware acceleration)")
    print("=" * 65)


def sample_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    """
    Uniform Temporal Subsampling:
      Selects `num_frames` evenly spaced frame indices across the full video duration.
      
    Mathematical Intuition (Lecture Slide 14-16: Training on Clips):
      Videos are typically recorded at 24-60 fps. Passing every single frame into
      a neural network would blow up computation and VRAM. Instead, models sample
      a small number of frames (e.g., T=8 or T=16) with uniform temporal stride.
    """
    if total_frames <= num_frames:
        return np.arange(total_frames)
    return np.linspace(0, total_frames - 1, num=num_frames, dtype=int)


def read_video_pyav(video_path: str, num_frames: int = 16, fast_seek: bool = True) -> np.ndarray:
    """
    Decodes exactly `num_frames` from an MP4/MKV video container using PyAV (FFmpeg).
    
    Why this is an Industrial Best Practice:
      1. Multi-threaded decoding: `stream.thread_type = 'AUTO'` engages multi-core
         decoding for H.264 / HEVC video streams.
      2. Timestamp Seeking: Instead of sequentially iterating through hundreds of frames
         (which is slow and incurs Python GIL overhead), we seek directly to target
         Presentation Timestamps (PTS) where keyframes (I-frames) reside.
         
    Returns:
      np.ndarray of shape [T, H, W, 3] in RGB format (uint8, range 0..255).
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    # Enable FFmpeg slice and frame multi-threading across all available CPU cores
    stream.thread_type = "AUTO"

    total_frames = stream.frames or 0
    duration = stream.duration

    # =========================================================================
    # OPTION A (Fast Path): Timestamp Seeking (Industrial Standard)
    # =========================================================================
    if fast_seek and duration is not None and duration > 0:
        # Calculate target presentation timestamps evenly spaced across the duration
        pts_targets = np.linspace(0, duration, num_frames, endpoint=False)
        frames = []
        for pts in pts_targets:
            # any_frame=False forces FFmpeg to seek to the nearest keyframe (I-frame)
            container.seek(int(pts), any_frame=False, stream=stream)
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))
                break  # Take only the first decoded frame after the seek
        container.close()
        if len(frames) == num_frames:
            return np.stack(frames)
        # If seek returned fewer frames than expected, fallback to linear decode below

    # =========================================================================
    # OPTION B (Fallback): Uniform Linear Scan
    # =========================================================================
    indices = set(sample_frame_indices(max(1, total_frames), num_frames))
    frames = []
    container.seek(0)
    for idx, frame in enumerate(container.decode(stream)):
        if idx in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) == num_frames:
            break
    container.close()

    if len(frames) == 0:
        raise ValueError(f"Could not decode any valid video frames from {video_path}")

    # Pad if video is shorter than requested num_frames
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.stack(frames[:num_frames])
