# IF5152: Advanced Computer Vision & Deep Learning
### Institut Teknologi Bandung (ITB)

Welcome to the course repository for **IF5152 (Advanced Computer Vision)**. This repository houses lecture reference materials, hands-on laboratory modules, and practical industrial implementations.

---

## 📚 Course Modules & Laboratories

| Module / Directory | Topic | Key Concepts | Hardware Support |
| :--- | :--- | :--- | :--- |
| **[`generative-models/`](./generative-models)** | **Generative Models (Lecture 13 & 14)** | Exact Likelihood (PixelCNN), Latent Failure Modes (Vanilla AE), ELBO & Reparameterization (VAE), Minimax Game (DCGAN), Iterative Denoising (DDPM) | Apple Silicon (MPS), NVIDIA GPU (CUDA), CPU |
| **[`video-understanding-lab/`](./video-understanding-lab)** | **Video Understanding (Week 11)** | Video I/O bottlenecks, Late Fusion, 3D Spatio-Temporal CNNs (R3D, R(2+1)D, MC3), Video Transformers (VideoMAE), Zero-Shot CLIP Search, Local M4 MPS Training | Apple Silicon (MPS), NVIDIA GPU (CUDA), CPU |

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

# 4. Session 1: Video Tensors, I/O Bottlenecks & Fusion Paradigms
python 01_video_io_and_fusion_paradigms.py --device mps

# 5. Session 2: Spatio-Temporal 3D CNNs & Factorization Benchmark
python 02_spatiotemporal_3d_cnns.py --model all --device mps

# 6. Session 3: Spatio-Temporal Vision Transformers (VideoMAE)
python 03_video_transformer_inference.py --device mps

# 7. Session 4: Open-Vocabulary Multimodal Video Search (CLIP)
python 04_zeroshot_video_search.py --device mps

# 8. Session 5: Local Training & Fine-Tuning on Apple M4 MPS
python 05_train_temporal_head.py --device mps --epochs 20
```

Refer to [`video-understanding-lab/README.md`](./video-understanding-lab/README.md) and the [`video-understanding-lab/docs/`](./video-understanding-lab/docs) directory for detailed self-study learning guides, theoretical derivations, and tensor dimensions.
