# IF5152: Advanced Computer Vision & Deep Learning
### Institut Teknologi Bandung (ITB)

Welcome to the course repository for **IF5152 (Advanced Computer Vision)**. This repository houses lecture reference materials, hands-on laboratory modules, and practical industrial implementations.

---

## 📚 Course Modules & Laboratories

| Module / Directory | Topic | Key Concepts | Hardware Support |
| :--- | :--- | :--- | :--- |
| **[`video-understanding-lab/`](./video-understanding-lab)** | **Video Understanding (Week 11)** | Video I/O bottlenecks, Late Fusion, Spatio-Temporal Transformers (VideoMAE), Zero-Shot Video Search (CLIP) | Apple Silicon (MPS), NVIDIA GPU (CUDA), CPU |

---

## 🛠️ General Environment Setup

All modules are designed to run seamlessly on:
- **Local Apple Silicon (M1/M2/M3/M4)** via Metal Performance Shaders (`mps`)
- **Cloud / Google Colab / Linux Workstations** via NVIDIA CUDA (`cuda`)
- **CPU Fallback** (`cpu`)

### Quickstart for `video-understanding-lab`:
```bash
# 1. Activate conda environment
conda activate video-understanding

# 2. Navigate to module
cd video-understanding-lab

# 3. Download demo media
python download_samples.py

# 4. Run Session 1 Demo (I/O & Late Fusion)
python 01_video_io_and_late_fusion.py --device mps

# 5. Run Session 2 Demo (Video Transformer & Zero-Shot Search)
python 02_video_transformer_inference.py --device mps
python 03_zeroshot_video_search.py --device mps
```

Refer to [`video-understanding-lab/README.md`](./video-understanding-lab/README.md) for detailed pedagogical notes, tensor dimensions, and slide mappings.
