# Session 3: Training & Fine-Tuning a Late Fusion Temporal Head on Apple Silicon M4 (MPS)

- **File**: [`04_train_late_fusion_head.py`](../04_train_late_fusion_head.py)
- **Lecture Alignment**: IF5152 Week 11 (Late Fusion Head Training & Optimization)
- **Target Hardware**: Apple Silicon (`mps`), NVIDIA Cloud GPU (`cuda`), or CPU fallback (`cpu`)

---

## 1. Pedagogical Objective

In this lab session, you will train a deep video action recognition model locally on your own machine:
1. **The Industrial Reality of Video Training**: Why training 3D CNNs or Video Transformers from scratch is impractical for edge devices/laptops, and how **feature caching + temporal head fine-tuning** provides an ultra-fast, industrial-grade training paradigm.
2. **Backpropagation Through Time (BPTT)**: How gradients propagate backwards through a recurrent **Gated Recurrent Unit (GRU)** temporal head across video frames.
3. **Apple Silicon M4 MPS Acceleration**: Leveraging PyTorch Metal Performance Shaders and Unified Memory Architecture (UMA) to train models in **fractions of a second per epoch**.

---

## 2. Theoretical Background & Mathematical Foundations

### The Computational Bottleneck in Video Deep Learning
When training on video datasets end-to-end, two major bottlenecks arise:
1. **CPU Video Decoding Bottleneck**: Decompressing H.264/MPEG video frames from disk repeatedly during training epochs consumes up to 90% of total training wall-clock time.
2. **Activation Memory Footprint**: Keeping intermediate feature maps for $T$ frames across all 2D/3D convolutional layers in VRAM during backpropagation quickly causes Out-Of-Memory (OOM) errors.

### The Solution: Feature Caching + Temporal Transfer Learning
Because the spatial filters of a 2D backbone pretrained on ImageNet (such as MobileNetV3 or ResNet) already capture low-level textures, edges, and object semantics, we can **freeze the spatial backbone** $\theta_{\text{backbone}}$:
$$\nabla_{\theta_{\text{backbone}}} \mathcal{L} = 0$$

```mermaid
flowchart TD
    subgraph OfflineCaching ["Step 1: One-Time Offline Feature Extraction (MPS)"]
        V["Dataset Video Clips [N clips]"] --> Dec["PyAV Fast Keyframe Decode<br/>[T=16 frames]"]
        Dec --> BB["Frozen 2D CNN Backbone<br/>(MobileNetV3 on Apple M4 MPS)"]
        BB --> Cache["Cached Feature Tensor<br/>Shape: [N, T=16, D=576] in RAM"]
    end

    subgraph TrainingLoop ["Step 2: Ultra-Fast Temporal Head Training (MPS)"]
        Cache --> Batch["Mini-batch [B=8, 16, 576]"]
        Batch --> Head["Trainable Head (GRU or Mean)"]
        Head --> Loss["CrossEntropyLoss"]
        Loss --> Opt["AdamW Optimizer (BPTT)<br/>~30ms / epoch on M4!"]
    end
```

### Mathematical Formulation of Temporal Heads

#### Option A: Recurrent Gated Recurrent Unit (GRU) Head
Given a cached sequence of spatial embeddings $\mathbf{h}_{1:T} = [\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_T] \in \mathbb{R}^{T \times D}$:
$$\mathbf{z}_t = \sigma(\mathbf{W}_z \mathbf{h}_t + \mathbf{U}_z \mathbf{s}_{t-1} + \mathbf{b}_z) \quad \text{(Update Gate)}$$
$$\mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{h}_t + \mathbf{U}_r \mathbf{s}_{t-1} + \mathbf{b}_r) \quad \text{(Reset Gate)}$$
$$\tilde{\mathbf{s}}_t = \tanh(\mathbf{W}_h \mathbf{h}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{s}_{t-1}) + \mathbf{b}_h)$$
$$\mathbf{s}_t = (1 - \mathbf{z}_t) \odot \mathbf{s}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{s}}_t$$

We pool the bidirectional hidden states across time and project to action class logits:
$$\hat{\mathbf{y}} = \mathbf{W}_{\text{head}} \left(\frac{1}{T}\sum_{t=1}^T \mathbf{s}_t\right) + \mathbf{b}_{\text{head}} \in \mathbb{R}^K$$

#### Option B: Temporal Mean Pooling Head
$$\bar{\mathbf{h}} = \frac{1}{T}\sum_{t=1}^T \mathbf{h}_t, \quad \hat{\mathbf{y}} = \mathbf{W}_{\text{head}} \bar{\mathbf{h}} + \mathbf{b}_{\text{head}}$$
This baseline has zero recurrent parameters and serves as an order-invariant ablation benchmark.

---

## 3. Tensor Dimension Tracing

| Pipeline Stage | Operation | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Video Decoding** | Keyframe Sampling | Disk Clip | `(16, 240, 320, 3)` | 16 RGB frames |
| **Preprocessing** | Resizing & Norm | `16 x [H, W, 3]` | `[16, 3, 224, 224]` | ImageNet mean/std |
| **Backbone Forward** | MobileNetV3 (Frozen) | `[16, 3, 224, 224]` | `[16, 576]` | Spatial pooling inside backbone |
| **Feature Caching** | Tensor Stack | $N$ clips | `[N, 16, 576]` | Cached in memory (~3.6 MB for 20 clips) |
| **Training Batch** | DataLoader | `[N, 16, 576]` | `[8, 16, 576]` | Batch size $B=8$ |
| **Bi-GRU Layer** | Recurrent Dynamics | `[8, 16, 576]` | `[8, 16, 256]` | Hidden dim 128 $\times 2$ directions |
| **Temporal Mean** | Sequence Pooling | `[8, 16, 256]` | `[8, 256]` | Compact video representation |
| **Classification Head** | Linear Projection | `[8, 256]` | `[8, 5]` | 5 action class logits |

---

## 4. Execution Playbook

### 1. Fast Training on Apple Silicon M4 (Default: UCF-101 5-Class Subset)
```bash
python 04_train_late_fusion_head.py --device mps --epochs 20 --fusion gru
```
*Automatically streams and extracts 5 classes (`ApplyEyeMakeup`, `BaseballPitch`, `Basketball`, `BenchPress`, `BandMarching`) on first run.*

### 2. Ablation: Comparing with Temporal Mean Pooling Baseline
```bash
python 04_train_late_fusion_head.py --device mps --epochs 20 --fusion mean
```

### 3. 100% Offline Test: Synthetic Motion Generator (Zero Download)
If running without internet access, use the built-in synthetic motion dataset generator:
```bash
python 04_train_late_fusion_head.py --dataset synthetic --device mps --epochs 10
```

### CLI Arguments Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `str` | `ucf_subset` | Dataset source: `ucf_subset` or `synthetic` |
| `--fusion` | `str` | `gru` | Head architecture: `gru` (recurrent) or `mean` (pooling) |
| `--epochs` | `int` | `20` | Total training epochs |
| `--batch_size` | `int` | `8` | Mini-batch size for temporal training |
| `--lr` | `float` | `0.001` | Learning rate for AdamW |
| `--hidden_dim` | `int` | `128` | Hidden units in GRU head |
| `--num_frames` | `int` | `16` | Sampled frames per video clip |
| `--clips_per_class` | `int` | `12` | Number of video clips per action class |
| `--device` | `str` | `auto` | Target accelerator (`auto`, `mps`, `cuda`, `cpu`) |

---

## 5. Expected Output & Interpretation

Running on Apple Silicon M4 with Metal Performance Shaders (`mps`):

```text
=================================================================
 PyTorch Version : 2.10.0
 Active Device   : MPS
 Backend         : Apple Silicon Metal Performance Shaders (MPS)
 Note            : Leveraging Unified Memory Architecture (UMA)
=================================================================

--- [STEP 1] Preparing Dataset (UCF_SUBSET) ---
Dataset Ready: 40 Train clips, 20 Val clips across 5 classes:
  Class #0: ApplyEyeMakeup
  Class #1: BandMarching
  Class #2: BaseballPitch
  Class #3: Basketball
  Class #4: BenchPress

--- [STEP 2] Precomputing Spatial Features on MPS ---
Loading pretrained MobileNetV3-Small spatial backbone...
Feature cache constructed: (40, 16, 576) in 1.42s (35.5 ms/clip)
Feature cache constructed: (20, 16, 576) in 0.65s (32.5 ms/clip)

--- [STEP 3] Initializing Trainable Temporal Head (GRU) ---
Trainable Parameters in Temporal Head: 543,493
Frozen Backbone Parameters             : 927,008

--- [STEP 4] Training on MPS (20 Epochs) ---
====================================================================
Epoch    Train Loss   Train Acc    Val Loss     Val Acc      Time    
====================================================================
01/20   1.6142       35.0%        1.3210       60.0%        38.2ms ★ (Best)
02/20   1.0210       85.0%        1.0540       75.0%        32.1ms ★ (Best)
05/20   0.3120       100.0%       0.6840       85.0%        31.5ms ★ (Best)
10/20   0.1140       100.0%       0.5210       95.0%        32.0ms ★ (Best)
20/20   0.0450       100.0%       0.4810       95.0%        31.8ms
====================================================================
Training completed in 0.85 seconds (32.1 ms/epoch average).
Best Validation Accuracy: 95.0%
Saved best model checkpoint to: 'checkpoints/best_head_gru.pt'

--- [STEP 5] Final Classification Summary ---
Overall Validation Accuracy: 95.00%

Class Name                Correct / Total    Accuracy
-------------------------------------------------------
  ApplyEyeMakeup          : 4/4            100.0%  ████████████████████
  BandMarching            : 4/4            100.0%  ████████████████████
  BaseballPitch           : 3/4             75.0%  ███████████████
  Basketball              : 4/4            100.0%  ████████████████████
  BenchPress              : 4/4            100.0%  ████████████████████
-------------------------------------------------------

[SUCCESS] Session 3 (Training Late Fusion Head) completed.
```

### Key Performance Takeaways:
1. **Unrivaled Iteration Speed**: Notice that each training epoch executes in approximately **~30–35 milliseconds**. The entire 20-epoch training run finishes in under 2 seconds!
2. **Unified Memory Efficiency**: Caching features directly on the M4 unified memory bus completely eliminates disk I/O bottlenecks and allows rapid experimentation with different temporal heads.

---

## 6. Self-Study Exercises & Experiments

### Exercise 1: GRU vs Mean Pooling Ablation
Run both models on the same split:
```bash
python 04_train_late_fusion_head.py --fusion gru --epochs 20
python 04_train_late_fusion_head.py --fusion mean --epochs 20
```
- Compare the final validation accuracy and convergence speed.
- On actions with distinct sequential stages (like `BaseballPitch` or `Basketball`), does the GRU achieve lower validation loss than Average Pooling?

### Exercise 2: Directional Sensitivity on Synthetic Motions
Run training on the synthetic motion dataset:
```bash
python 04_train_late_fusion_head.py --dataset synthetic --fusion gru --epochs 10
```
- Inspect how well the model separates `drift_right` from `drift_down` and `rotate_clockwise`.
- Try inverting the temporal order of validation features (`features = torch.flip(features, dims=[1])`). Does the GRU detect the reversal?

---

## 7. Troubleshooting FAQ

- **Error: `ModuleNotFoundError: No module named 'cv2'`**:
  - Install OpenCV via `pip install opencv-python-headless`.
- **Warning: `UserWarning: The operator 'aten::gru' is not supported on MPS...`**:
  - In modern PyTorch ($\ge 2.0$), GRU has full native MPS support. If you are on an older PyTorch build, export `PYTORCH_ENABLE_MPS_FALLBACK=1` in your terminal.
- **Checkpoints directory**:
  - Best model weights are automatically saved to `checkpoints/best_head_<fusion>.pt`. You can load them later with `torch.load()`.
