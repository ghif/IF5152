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
IF5152/
├── data/                             # Shared course data directory (root level)
└── video-understanding-lab/          # Week 11 Module
    ├── docs/                         # 📖 In-depth educational notes & self-study guides
    │   ├── 01_video_io_and_fusion_paradigms.md
    │   ├── 02_spatiotemporal_3d_cnns.md
    │   ├── 03_video_transformer_inference.md
    │   ├── 04_zeroshot_video_search.md
    │   └── 05_train_temporal_head.md
    ├── 01_video_io_and_fusion_paradigms.py # Session 1: Video I/O bottleneck & Fusion Paradigms
    ├── 02_spatiotemporal_3d_cnns.py      # Session 2: Spatio-Temporal 3D CNNs (R3D, R(2+1)D, MC3)
    ├── 03_video_transformer_inference.py # Session 3: Pretrained VideoMAE (Kinetics-400)
    ├── 04_zeroshot_video_search.py       # Session 4: Open-Vocabulary Video-Language Matching (CLIP)
    ├── 05_train_temporal_head.py         # Session 5: Train / Fine-tune Temporal Head on Apple M4 MPS
    ├── download_samples.py               # Helper script to fetch sample video clips into data/
    ├── utils.py                          # Device management & PyAV fast decoder
    ├── environment.yml                   # Conda environment definition
    ├── requirements.txt                  # Pip dependencies
    └── README.md                         # Module documentation
```

---

## 📖 Educational Notes & Self-Study Guides

To enable students to learn autonomously, run the code independently, and deeply understand the theoretical and mathematical foundations, each script is accompanied by a dedicated Markdown companion guide:

| Script | Companion Guide | Core Topics | Key Concepts Covered |
| :--- | :--- | :--- | :--- |
| [`01_video_io_and_fusion_paradigms.py`](01_video_io_and_fusion_paradigms.py) | [**`docs/01_video_io_and_fusion_paradigms.md`**](docs/01_video_io_and_fusion_paradigms.md) | Video Tensors & Fusion Paradigms | 5D Video Tensors (`[B, T, C, H, W]`), PyAV Keyframe Seeking benchmark, Early Fusion vs Late Fusion (Mean Pooling & Recurrent GRU) |
| [`02_spatiotemporal_3d_cnns.py`](02_spatiotemporal_3d_cnns.py) | [**`docs/02_spatiotemporal_3d_cnns.md`**](docs/02_spatiotemporal_3d_cnns.md) | Spatio-Temporal 3D CNNs & Factorization | 3D Volumetric Convolutions, $(2+1)\text{D}$ Spatial-Temporal Factorization, Mixed Convolutions (MC3), I3D Weight Inflation, `[B, C, T, H, W]` layout |
| [`03_video_transformer_inference.py`](03_video_transformer_inference.py) | [**`docs/03_video_transformer_inference.md`**](docs/03_video_transformer_inference.md) | Spatio-Temporal Attention & VideoMAE | Quadratic complexity of Space-Time Attention, Divided Attention (TimeSformer/VideoMAE), 3D Tubelet embeddings, Kinetics-400 |
| [`04_zeroshot_video_search.py`](04_zeroshot_video_search.py) | [**`docs/04_zeroshot_video_search.md`**](docs/04_zeroshot_video_search.md) | Multimodal Video-Language & CLIP | Closed-set limits vs Open-vocabulary search, Dual-encoder multimodal matching (CLIP), Unit hypersphere normalization, Vector DB search |
| [`05_train_temporal_head.py`](05_train_temporal_head.py) | [**`docs/05_train_temporal_head.md`**](docs/05_train_temporal_head.md) | Temporal Head Training & Optimization | Local training on Apple Silicon M4 MPS, Feature caching, Backpropagation Through Time (BPTT), Recurrent GRU vs Mean Pooling |

*Each companion guide includes theoretical derivations, Mermaid architecture flowcharts, tensor dimension tracing tables, execution playbooks, expected outputs, self-study exercises, and troubleshooting tips.*

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

### Session 1: Video Tensors, The I/O Bottleneck & Fusion Paradigms
* **Core Topics**: Video Fundamentals, Single-Frame CNN, Early Fusion vs Late Fusion (Mean Pooling & Recurrent GRU).
* **Self-Study Guide**: 📖 [Detailed Notes & Theory Guide](docs/01_video_io_and_fusion_paradigms.md)
* **Industrial Reality**: Decoding video frames is a major CPU-bound bottleneck in production pipelines. We benchmark decoding speedups (PyAV fast seek) and compare Early Fusion (pixel-level channel stacking) against Late Fusion (embedding-level temporal pooling & recurrent GRU).
* **Run Demo**:
  ```bash
  # Head-to-Head Comparative Benchmark across all paradigms:
  python 01_video_io_and_fusion_paradigms.py --fusion all --device mps

  # Early Fusion (channel stacking into inflated Conv2d stem)
  python 01_video_io_and_fusion_paradigms.py --fusion early

  # Late Fusion (temporal mean pooling)
  python 01_video_io_and_fusion_paradigms.py --fusion mean

  # Late Fusion (recurrent GRU)
  python 01_video_io_and_fusion_paradigms.py --fusion gru

  # Explicit device selection (auto | mps | cuda | cpu)
  python 01_video_io_and_fusion_paradigms.py --device mps
  ```

---

### Session 2: Spatio-Temporal 3D CNNs & Architectural Factorization
* **Core Topics**: Full 3D Convolutions ($R3D$), $(2+1)\text{D}$ Spatial-Temporal Factorization ($R(2+1)D$), Mixed Convolutions ($MC3$), I3D Weight Inflation.
* **Self-Study Guide**: 📖 [3D CNNs & Architectural Factorization Guide](docs/02_spatiotemporal_3d_cnns.md)
* **Industrial Reality**: 3D convolutions capture joint space-time patterns (optical flow, trajectories) directly in early layers. Factorizing $3 \times 3 \times 3$ into $(1 \times 3 \times 3) + (3 \times 1 \times 1)$ cuts parameters by over 50% and doubles non-linear activations (ReLU), making models easier to train and faster to converge.
* **Run Demos**:

  **1. Run Factorized R(2+1)D on Apple Silicon MPS:**
  ```bash
  python 02_spatiotemporal_3d_cnns.py --model r2plus1d --device mps
  ```

  **2. Inspect Layer Decompositions & Factorized Stems:**
  ```bash
  python 02_spatiotemporal_3d_cnns.py --model r2plus1d --print-model
  ```

  **3. Head-to-Head Architectural Benchmark (R3D vs R(2+1)D vs MC3):**
  ```bash
  python 02_spatiotemporal_3d_cnns.py --model all --device mps
  ```

---

### Session 3: Spatio-Temporal Vision Transformers (ViViT & VideoMAE)
* **Core Topics**: Non-Local Blocks, Space-Time Transformers, ViViT Transfer Learning (Model 2 Factorized Encoder), Divided Attention (TimeSformer), 3D Tubelet Embeddings, VideoMAE 90% Masked Autoencoding.
* **Self-Study Guide**: 📖 [Video Transformers: 2D ViT to ViViT & VideoMAE Guide](docs/03_video_transformer_inference.md)
* **Industrial Reality**: Training 3D video transformers from scratch is computationally prohibitive. Industry relies either on transfer learning from 2D ViT (ViViT) or foundation models pretrained at scale via self-supervised masked autoencoding (VideoMAE).
* **Run Demos**:
  ```bash
  # 1. Head-to-Head Comparative Benchmark (VideoMAE vs ViViT):
  python 03_video_transformer_inference.py --model all --device mps

  # 2. Run ViViT (32 frames):
  python 03_video_transformer_inference.py --model vivit --device mps

  # 3. Run VideoMAE (16 frames):
  python 03_video_transformer_inference.py --model videomae --device mps

  # 4. Inspect PyTorch Educational ViViT Factorized Architecture:
  python 03_video_transformer_inference.py --print-model
  ```

---

### Session 4: Open-Vocabulary Video Search with Multimodal Foundation Models (CLIP)
* **Core Topics**: Closed-Set Limits, Dual-Encoder Architectures, Joint Vision-Language Alignment, Hypersphere Projection.
* **Self-Study Guide**: 📖 [Zero-Shot Multimodal Search (CLIP) Guide](docs/04_zeroshot_video_search.md)
* **Industrial Reality**: Modern video search engines (YouTube, TikTok) query millions of untrimmed videos using natural language without requiring pre-defined category labels.
* **Run Demo**:
  ```bash
  python 04_zeroshot_video_search.py --device mps
  ```

---

### Session 5: Local Training & Fine-Tuning on Apple Silicon M4 (MPS)
* **Core Topics**: Backpropagation Through Time, Feature Caching, Temporal Head Optimization.
* **Self-Study Guide**: 📖 [Local Training & MPS Acceleration Guide](docs/05_train_temporal_head.md)
* **Industrial Reality**: Decompressing video frames repeatedly during training is a massive CPU bottleneck. We precompute spatial features on the M4 GPU once, allowing the recurrent GRU head to train at **~30 milliseconds per epoch**!
* **Run Training**:

  **1. Train Recurrent GRU on UCF-101 Mini Subset:**
  ```bash
  python 05_train_temporal_head.py --device mps --epochs 20 --fusion gru
  ```

  **2. Train Temporal Mean Pooling Baseline:**
  ```bash
  python 05_train_temporal_head.py --device mps --epochs 20 --fusion mean
  ```

  **3. 100% Offline Test with Synthetic Motion Generator:**
  ```bash
  python 05_train_temporal_head.py --dataset synthetic --device mps --epochs 10
  ```

---

## ☁️ Running on Google Colab CLI / Cloud GPU

You can run these exact same scripts inside a Google Colab terminal, remote VM, or Docker container with zero code modifications:

```bash
# Check GPU
nvidia-smi

# Run scripts with automatic CUDA detection
python 01_video_io_and_fusion_paradigms.py --device cuda
python 02_spatiotemporal_3d_cnns.py --model all --device cuda
python 03_video_transformer_inference.py --device cuda
python 04_zeroshot_video_search.py --device cuda
python 05_train_temporal_head.py --device cuda --epochs 20
```
