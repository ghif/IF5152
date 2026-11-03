# IF5152: Video Understanding Live Coding Lab
### Practical & Industrial Perspectives for Video Deep Learning

This repository contains ready-to-run, modular Python CLI scripts designed for teaching **Video Understanding** (IF5152 Week 11). The material bridges classical academic theory (CS231n Video Lecture reference) with modern industrial practice.

It is designed to run seamlessly across:
- **Apple Silicon (M1/M2/M3/M4)** via PyTorch Metal Performance Shaders (`mps`)
- **Cloud / Google Colab CLI** via NVIDIA GPUs (`cuda`)
- **Standard CPU** fallback (`cpu`)

---

## 📁 Repository Structure

```text
video-understanding-lab/
├── 01_video_io_and_late_fusion.py    # Session 1: Video I/O bottleneck & Late Fusion (Mean vs GRU)
├── 02_video_transformer_inference.py # Session 2: Pretrained VideoMAE (Kinetics-400)
├── 03_zeroshot_video_search.py       # Session 2: Open-Vocabulary Video-Language Matching (CLIP)
├── download_samples.py               # Helper script to fetch sample video clips
├── utils.py                          # Device management (MPS/CUDA/CPU) & PyAV decoder
├── environment.yml                   # Conda environment definition
├── requirements.txt                  # Pip dependencies
└── data/                             # Video clips directory
```

---

## 🚀 Environment Setup

### 1. Dedicated Conda Environment

To create and activate the environment:

```bash
# Using conda / mamba
conda env create -f environment.yml
conda activate video-understanding
```

Or install directly with pip:
```bash
pip install -r requirements.txt
```

### 2. Download Demo Video

Run the lightweight downloader to fetch a sample video clip into `data/`:

```bash
python download_samples.py
```

---

## 🎓 Curriculum & Live Demo Walkthrough

### Session 1: Video Tensors, The I/O Bottleneck & Late Fusion
* **Reference Slides**: Slides 1–55 (Video = 2D + Time, Single-Frame CNN, Early/Late Fusion, 3D Conv).
* **Industrial Reality**: Decoding video frames is a major CPU-bound bottleneck in production pipelines. Simple 2D backbone feature extraction with temporal pooling provides a strong, low-latency baseline.
* **Run Demo**:
  ```bash
  # Mean pooling (Slide 19)
  python 01_video_io_and_late_fusion.py --fusion mean

  # Recurrent temporal head (Slide 63)
  python 01_video_io_and_late_fusion.py --fusion gru

  # Explicit device selection (auto | mps | cuda | cpu)
  python 01_video_io_and_late_fusion.py --device mps
  ```

---

### Session 2: Video Transformers & Zero-Shot Video-Language Matching
* **Reference Slides**: Slides 56–105 (Non-Local Blocks, Space-Time Transformers, Untrimmed Video, Multimodal Audio-Visual).
* **Industrial Reality**: Training 3D models from scratch is rare in practice. Industry relies on Video Foundation Models (VideoMAE) and multimodal zero-shot retrieval (CLIP / X-CLIP) for open-vocabulary search and content moderation.
* **Run Demos**:

  **1. Pretrained Video Transformer (VideoMAE on Kinetics-400):**
  ```bash
  python 02_video_transformer_inference.py --device mps
  ```

  **2. Zero-Shot Video Action Search with CLIP:**
  ```bash
  python 03_zeroshot_video_search.py --device mps
  ```

---

## ☁️ Running on Google Colab CLI / Cloud GPU

You can run these exact same scripts inside a Google Colab terminal, remote VM, or Docker container with zero code modifications:

```bash
# Check GPU
nvidia-smi

# Run scripts with automatic CUDA detection
python 01_video_io_and_late_fusion.py --device cuda
python 02_video_transformer_inference.py --device cuda
python 03_zeroshot_video_search.py --device cuda
```
