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
    │   ├── 01_video_io_and_late_fusion.md
    │   ├── 02_video_transformer_inference.md
    │   ├── 03_zeroshot_video_search.md
    │   ├── 04_train_late_fusion_head.md
    │   └── 05_spatiotemporal_3d_cnns.md
    ├── 01_video_io_and_late_fusion.py    # Session 1: Video I/O bottleneck & Late Fusion (Mean vs GRU)
    ├── 02_video_transformer_inference.py # Session 2: Pretrained VideoMAE (Kinetics-400)
    ├── 03_zeroshot_video_search.py       # Session 2: Open-Vocabulary Video-Language Matching (CLIP)
    ├── 04_train_late_fusion_head.py      # Session 3: Train / Fine-tune Temporal Head on Apple M4 MPS
    ├── 05_spatiotemporal_3d_cnns.py      # Session 1B: Spatio-Temporal 3D CNNs (R3D, R(2+1)D, MC3)
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
| [`01_video_io_and_late_fusion.py`](01_video_io_and_late_fusion.py) | [**`docs/01_video_io_and_late_fusion.md`**](docs/01_video_io_and_late_fusion.md) | Video Tensors & Late Fusion | 5D Video Tensors (`[B, T, C, H, W]`), PyAV Keyframe Seeking benchmark, 2D CNN Backbones, Mean Pooling vs Recurrent GRU |
| [`02_video_transformer_inference.py`](02_video_transformer_inference.py) | [**`docs/02_video_transformer_inference.md`**](docs/02_video_transformer_inference.md) | Spatio-Temporal Attention & VideoMAE | Quadratic complexity of Space-Time Attention, Divided Attention (TimeSformer/VideoMAE), 3D Tubelet embeddings, Kinetics-400 |
| [`03_zeroshot_video_search.py`](03_zeroshot_video_search.py) | [**`docs/03_zeroshot_video_search.md`**](docs/03_zeroshot_video_search.md) | Multimodal Video-Language & CLIP | Closed-set limits vs Open-vocabulary search, Dual-encoder multimodal matching (CLIP), Unit hypersphere normalization, Vector DB search |
| [`04_train_late_fusion_head.py`](04_train_late_fusion_head.py) | [**`docs/04_train_late_fusion_head.md`**](docs/04_train_late_fusion_head.md) | Temporal Head Training & Optimization | Local training on Apple Silicon M4 MPS, Feature caching, Backpropagation Through Time (BPTT), Recurrent GRU vs Mean Pooling |
| [`05_spatiotemporal_3d_cnns.py`](05_spatiotemporal_3d_cnns.py) | [**`docs/05_spatiotemporal_3d_cnns.md`**](docs/05_spatiotemporal_3d_cnns.md) | Spatio-Temporal 3D CNNs & Factorization | 3D Volumetric Convolutions, $(2+1)\text{D}$ Spatial-Temporal Factorization, Mixed Convolutions (MC3), I3D Weight Inflation, `[B, C, T, H, W]` layout |

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

### Session 1: Video Tensors, The I/O Bottleneck & Late Fusion
* **Core Topics**: Video Fundamentals, Single-Frame CNN, Early/Late Fusion, 3D Conv.
* **Self-Study Guide**: 📖 [Detailed Notes & Theory Guide](docs/01_video_io_and_late_fusion.md)
* **Industrial Reality**: Decoding video frames is a major CPU-bound bottleneck in production pipelines. Simple 2D backbone feature extraction with temporal pooling provides a strong, low-latency baseline.
* **Run Demo**:
  ```bash
  # Mean pooling
  python 01_video_io_and_late_fusion.py --fusion mean

  # Recurrent temporal head
  python 01_video_io_and_late_fusion.py --fusion gru

  # Explicit device selection (auto | mps | cuda | cpu)
  python 01_video_io_and_late_fusion.py --device mps
  ```

---

### Session 1B: Spatio-Temporal 3D CNNs & Architectural Factorization
* **Core Topics**: Full 3D Convolutions ($R3D$), $(2+1)\text{D}$ Spatial-Temporal Factorization ($R(2+1)D$), Mixed Convolutions ($MC3$), I3D Weight Inflation.
* **Self-Study Guide**: 📖 [3D CNNs & Architectural Factorization Guide](docs/05_spatiotemporal_3d_cnns.md)
* **Industrial Reality**: 3D convolutions capture joint space-time patterns (optical flow, trajectories) directly in early layers. Factorizing $3 \times 3 \times 3$ into $(1 \times 3 \times 3) + (3 \times 1 \times 1)$ cuts parameters by over 50% and doubles non-linear activations (ReLU), making models easier to train and faster to converge.
* **Run Demos**:

  **1. Run Factorized R(2+1)D on Apple Silicon MPS:**
  ```bash
  python 05_spatiotemporal_3d_cnns.py --model r2plus1d --device mps
  ```

  **2. Inspect Layer Decompositions & Factorized Stems:**
  ```bash
  python 05_spatiotemporal_3d_cnns.py --model r2plus1d --print-model
  ```

  **3. Head-to-Head Architectural Benchmark (R3D vs R(2+1)D vs MC3):**
  ```bash
  python 05_spatiotemporal_3d_cnns.py --model all --device mps
  ```

---

### Session 2: Video Transformers & Zero-Shot Video-Language Matching
* **Core Topics**: Non-Local Blocks, Space-Time Transformers, Untrimmed Video, Multimodal Audio-Visual.
* **Self-Study Guides**:
  * 📖 [VideoMAE Pretrained Transformer Guide](docs/02_video_transformer_inference.md)
  * 📖 [Zero-Shot Multimodal Search (CLIP) Guide](docs/03_zeroshot_video_search.md)
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

### Session 3: Local Training & Fine-Tuning on Apple Silicon M4 (MPS)
* **Core Topics**: Backpropagation Through Time, Late Temporal Fusion Training.
* **Self-Study Guide**: 📖 [Local Training & MPS Acceleration Guide](docs/04_train_late_fusion_head.md)
* **Industrial Reality**: Decompressing video frames repeatedly during training is a massive CPU bottleneck. We precompute spatial features on the M4 GPU once, allowing the recurrent GRU head to train at **~30 milliseconds per epoch**!
* **Run Training**:

  **1. Train Recurrent GRU on UCF-101 Mini Subset:**
  ```bash
  python 04_train_late_fusion_head.py --device mps --epochs 20 --fusion gru
  ```

  **2. Train Temporal Mean Pooling Baseline:**
  ```bash
  python 04_train_late_fusion_head.py --device mps --epochs 20 --fusion mean
  ```

  **3. 100% Offline Test with Synthetic Motion Generator:**
  ```bash
  python 04_train_late_fusion_head.py --dataset synthetic --device mps --epochs 10
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
