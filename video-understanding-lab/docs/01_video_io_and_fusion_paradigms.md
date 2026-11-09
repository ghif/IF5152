# Session 1: Video I/O Bottlenecks & Spatio-Temporal Fusion Paradigms

- **File**: [`01_video_io_and_fusion_paradigms.py`](../01_video_io_and_fusion_paradigms.py)
- **Lecture Alignment**: IF5152 Week 11 (Video Understanding Foundations & Fusion Paradigms)
- **Target Hardware**: Apple Silicon (`mps`), NVIDIA Cloud GPU (`cuda`), or CPU fallback (`cpu`)

---

## 1. Pedagogical Objective

In this lab session, you will explore the transition from **2D static image classification** to **spatio-temporal video understanding**:
1. **The Video I/O Decoding Bottleneck**: Why traditional frame-by-frame decoding severely throttles deep learning training pipelines, and how multi-threaded keyframe seeking solves it.
2. **Video Compression Architecture**: How modern codecs (H.264/HEVC) compress video using Group of Pictures (GOP), I/P/B-frames, and presentation timestamps (PTS).
3. **5D Video Tensor Mechanics**: How video clips are structured in PyTorch as `[Batch, Time, Channels, Height, Width]`.
4. **Spatio-Temporal Fusion Paradigms (Karpathy et al., CVPR 2014)**: Exploring and benchmarking:
   - **Early Fusion**: Pixel-level channel stacking into a 2D CNN stem with Channel Weight Inflation.
   - **Late Fusion (Mean Pooling)**: Order-invariant average pooling over deep 2D backbone spatial representations.
   - **Late Fusion (Recurrent GRU)**: Order-sensitive temporal hidden state modeling with Gated Recurrent Units.

---

## 2. Theoretical Background: The Video Processing & I/O Mechanism

### 2.1 The Computational & Bandwidth Explosion
While a standard image is a 3D tensor $[C \times H \times W]$, a video clip introduces a temporal dimension $T$:

$$
\mathbf{X} \in \mathbb{R}^{B \times T \times C \times H \times W}
$$

Consider the bandwidth of uncompressed raw video:
* A standard $1080\text{p}$ video clip at 30 frames per second (FPS):

$$
\text{Size per frame} = 1920 \times 1080 \times 3 \text{ bytes (RGB24)} \approx 6.22 \text{ MB}
$$

$$
\text{Data throughput} = 6.22 \text{ MB} \times 30 \text{ frames/sec} \approx 186.6 \text{ MB/second}
$$

* A 10-minute video clip in uncompressed RGB format would consume over **111 Gigabytes** of disk space!

To make storage and internet streaming feasible, all real-world video files are heavily compressed using lossy video codecs (e.g., H.264/AVC ([Wiegand et al., IEEE TCSVT 2003](https://ieeexplore.ieee.org/document/1218189)), H.265/HEVC ([Sullivan et al., IEEE TCSVT 2012](https://ieeexplore.ieee.org/document/6316136)), VP9, AV1).

---

### 2.2 Video Containers vs. Video Codecs
Students often confuse containers with codecs:
* **Container Format (`.mp4`, `.mkv`, `.avi`)**: A file wrapper (multiplexer) containing metadata (frame rate, duration, audio/video synchronization tracks, and index tables like the MP4 `moov` atom).
* **Video Codec (H.264, HEVC, ProRes)**: The actual compression algorithm that encodes raw pixel buffers into compressed bitstream packets.

---

### 2.3 Inter-Frame Compression & The Group of Pictures (GOP)
Video codecs exploit two types of redundancies:
1. **Spatial Redundancy (Intra-frame)**: Adjacent pixels within the same image often share color and texture (compressed via Discrete Cosine Transform / DCT and quantization).
2. **Temporal Redundancy (Inter-frame)**: Consecutive frames in a 30 FPS video are nearly identical, with only small regions moving.

Codecs organize video sequences into a repeating **Group of Pictures (GOP)** consisting of three frame types:

```mermaid
flowchart LR
    subgraph GOP ["Group of Pictures (GOP)"]
        direction LR
        I1["I-Frame<br/>(Keyframe)<br/>Self-Contained"]
        B1["B-Frame<br/>(Bi-directional)"]
        B2["B-Frame<br/>(Bi-directional)"]
        P1["P-Frame<br/>(Predicted)"]
        B3["B-Frame<br/>(Bi-directional)"]
        P2["P-Frame<br/>(Predicted)"]
        I2["I-Frame<br/>(Next Keyframe)"]

        I1 -->|Forward Prediction| P1
        P1 -->|Forward Prediction| P2
        I1 -.->|Interpolation| B1
        P1 -.->|Interpolation| B1
        I1 -.->|Interpolation| B2
        P1 -.->|Interpolation| B2
        P1 -.->|Interpolation| B3
        P2 -.->|Interpolation| B3
        P2 --> I2
    end
```

| Frame Type | Full Name | Compression Mechanism | Decoding Dependencies | Size Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **I-Frame** | Intra-coded (Keyframe) | Encoded independently like a standalone JPEG image using spatial DCT. | **None**. Can be decoded instantly in isolation. | **Largest** (100%) |
| **P-Frame** | Predicted Frame | Stores only **motion vectors** and residual error relative to the preceding I-frame or P-frame. | Requires decoding all preceding reference frames in the GOP. | **Medium** (~20–30%) |
| **B-Frame** | Bi-directional Frame | Stores motion vectors interpolated from both **preceding and succeeding** reference frames. | Requires decoding both past and future anchor frames. | **Smallest** (~5–10%) |

#### Decode Time Stamp (DTS) vs. Presentation Time Stamp (PTS):
Because B-frames require future reference frames to decode, video codecs transmit frames in **Decode Order (DTS)** which differs from their human display order **Presentation Order (PTS)**:
* **Display Order (PTS)**: $I_0 \rightarrow B_1 \rightarrow B_2 \rightarrow P_3$
* **Storage/Decode Order (DTS)**: $I_0 \rightarrow P_3 \rightarrow B_1 \rightarrow B_2$  *(The decoder must reconstruct $P_3$ before it can render $B_1$ and $B_2$!)*

---

### 2.4 Why Video I/O is the #1 Bottleneck in Deep Learning
When training neural networks on image datasets (e.g., ImageNet, [Deng et al., CVPR 2009](https://ieeexplore.ieee.org/document/5206848)), reading a JPEG image from disk takes $\sim 1\text{ ms}$.
However, when training on video clips:
* An epoch requires sampling thousands of video clips.
* If a model needs 16 frames uniformly spaced across a 600-frame video, how should those frames be retrieved?
* If a naive loop is used, the CPU must decompress entropy blocks (CABAC), compute inverse transforms, and apply motion compensation across **all 600 frames** just to throw away 584 of them!
* While the CPU spends $150\text{--}300\text{ ms}$ decoding frames, the GPU/MPS accelerator (which can compute a forward pass in $5\text{--}15\text{ ms}$) sits completely idle. This is known as **GPU Starvation**.

---

### 2.5 Algorithmic Comparison of Decoding Approaches

`01_video_io_and_fusion_paradigms.py` implements a live benchmark comparing three distinct decoding algorithms on the exact same video file:

```mermaid
flowchart TD
    subgraph Algo1 ["Algorithm 1: OpenCV Sequential Loop"]
        O1["Open Video File"] --> O2["Sequential cap.read() Loop"]
        O2 --> O3["Linear decode of every packet (0 to N)"]
        O3 --> O4["Discard frames not in sample index"]
        O4 --> O5["High C++ decode time across all 600+ frames"]
    end

    subgraph Algo2 ["Algorithm 2: PyAV Naive Linear Decode"]
        P1["Open Container via PyAV"] --> P2["for frame in container.decode()"]
        P2 --> P3["Decompress every packet (0 to N)"]
        P3 --> P4["Allocate high-level Python 'av.VideoFrame' wrapper"]
        P4 --> P5["Severe Python GIL overhead and GC pressure"]
    end

    subgraph Algo3 ["Algorithm 3: Industrial Fast Seek (PyAV / FFmpeg)"]
        F1["Inspect MP4 Metadata / moov Atom Index"] --> F2["Calculate exact PTS targets for 16 frames"]
        F3["container.seek(pts, any_frame=False)"]
        F4["Jump file pointer directly to nearest Keyframe (I-frame)"]
        F5["stream.thread_type = 'AUTO' (Multi-core slice decode)"]
        F6["3x to 5x faster; only touches target packets!"]
        F3 --> F4 --> F5 --> F6
    end
```

#### 1. Naive Sequential Decoding (`OpenCV`):
* Calls `cap.read()` in a linear loop starting from frame 0.
* Time complexity: $\mathcal{O}(N_{\text{total}})$, where $N_{\text{total}}$ is the total number of frames in the video.
* To extract 16 frames from a 647-frame video, it must decode **all 647 frames**.

#### 2. Naive Linear Decode with High-Level Python Wrappers (`PyAV Naive`):
* Iterates linearly using Python: `for frame in container.decode(stream): ...`
* In addition to decoding all 647 frames, PyAV instantiates a rich Python `av.VideoFrame` wrapper object for **every single discarded frame**.
* In Python, performing 647 object allocations, reference counts, and garbage collections under the Global Interpreter Lock (GIL) makes this approach even slower than OpenCV!

#### 3. Industrial Fast Seek via Demuxer Keyframe Navigation (`PyAV Fast Seek`):
* Reads the MP4 container's index table to find the byte offset corresponding to the nearest preceding **I-frame** for each target Presentation Time Stamp (PTS):

$$
\text{pts\_targets} = \text{linspace}(0, \text{duration}, T)
$$

* Executes `container.seek(pts, any_frame=False)` to jump the underlying disk file pointer directly to that byte offset, completely bypassing intermediate compressed packets.
* Enables hardware slice/frame multithreading:
  ```python
  stream.thread_type = "AUTO"  # Spawns worker threads across all available CPU cores
  ```
* Time complexity: $\mathcal{O}(T \cdot L_{\text{GOP}})$, where $T$ is the number of sampled frames ($16$) and $L_{\text{GOP}}$ is the distance to the keyframe ($\le 12\text{--}30$ frames).
* **Result**: Achieves **3x to 5x speedups** over sequential loops, keeping GPU/MPS tensor cores fully saturated.

---

### 2.6 The End-to-End Video Preprocessing Pipeline

Once raw frames are decoded, they undergo a multi-stage transformation into a normalized 5D PyTorch tensor:

```mermaid
flowchart TD
    A["Raw Compressed Packets (Disk)"] -->|"FFmpeg Demux + Decode"| B["YUV420p Planar Arrays"]
    B -->|"Color Conversion (sws_scale)"| C["RGB24 NumPy Array (T, H, W, 3)<br/>uint8: range 0 to 255"]
    C -->|"Bilinear Spatial Resizing"| D["Resized Frames (T, 224, 224, 3)"]
    D -->|"Scale to 0.0 - 1.0 (ToTensor)"| E["Float32 Tensor (T, 3, 224, 224)"]
    E -->|"ImageNet Standardization"| F["Normalized Tensor (T, 3, 224, 224)"]
    F -->|"Add Batch Dimension (unsqueeze 0)"| G["Batched 5D Video Tensor<br/>Shape: (B=1, T=16, C=3, H=224, W=224)"]
```

#### Mathematical Steps in Preprocessing:
1. **Color Space Conversion**: Video codecs compress video in **YUV420p** (luma/chroma subsampling). The decoder converts this to **RGB24** using the ITU-R BT.601 (full-range JFIF) color conversion matrix (or ITU-R BT.709 for HD content):

$$
\begin{bmatrix} R \\ G \\ B \end{bmatrix} = \begin{bmatrix} 1 & 0 & 1.402 \\ 1 & -0.344136 & -0.714136 \\ 1 & 1.772 & 0 \end{bmatrix} \begin{bmatrix} Y \\ U - 128 \\ V - 128 \end{bmatrix}
$$

2. **Spatial Resizing**: Resizes spatial dimensions from arbitrary video resolution $(H_{\text{orig}}, W_{\text{orig}})$ to the model input resolution $(224 \times 224)$ via bilinear interpolation.
3. **Channel Standardization**: Re-scales pixel values to $[0.0, 1.0]$ and applies standard ImageNet mean and standard deviation:

$$
x_{\text{norm}}^{(c)} = \frac{x^{(c)} - \mu^{(c)}}{\sigma^{(c)}}, \quad \mu = [0.485, 0.456, 0.406], \; \sigma = [0.229, 0.224, 0.225]
$$

---

## 3. Spatio-Temporal Fusion Paradigms (Karpathy et al., CVPR 2014)

In their seminal work *"Large-scale Video Classification with Convolutional Neural Networks"*, [Karpathy et al. (CVPR 2014)](https://cs.stanford.edu/people/karpathy/deepvideo/) systematically investigated where and how temporal information should be merged when processing video through 2D convolutional networks.

```mermaid
flowchart TD
    subgraph EarlyFusion ["Paradigm 1: Early Fusion (Pixel-Level)"]
        V1["Video Clip: [B, T, C, H, W]"] --> Stk1["Stack Frames along Channels<br/>[B, T*C, H, W] (e.g. 48 channels)"]
        Stk1 --> Conv1["Modified Stem Conv2D<br/>in_channels = T * C = 48<br/>Weights Inflated: W_2D / T"]
        Conv1 --> BB1["Standard 2D Backbone<br/>(1 forward pass for the whole clip!)"]
        BB1 --> H1["Linear Head -> Classes"]
    end

    subgraph LateFusionMean ["Paradigm 2: Late Fusion (Temporal Mean)"]
        V2["Video Clip: [B, T, C, H, W]"] --> Fl2["Fold Time into Batch<br/>[B*T, C, H, W] (16 frames)"]
        Fl2 --> BB2["Standard 2D Backbone<br/>(16 parallel forward passes)"]
        BB2 --> Feat2["Frame Features: [B, T, 576]"]
        Feat2 --> Pool2["Temporal Average Pooling<br/>[B, 576] (Order-invariant)"]
        Pool2 --> H2["Linear Head -> Classes"]
    end

    subgraph LateFusionGRU ["Paradigm 3: Late Fusion (Recurrent GRU)"]
        V3["Video Clip: [B, T, C, H, W]"] --> Fl3["Fold Time into Batch<br/>[B*T, C, H, W]"]
        Fl3 --> BB3["Standard 2D Backbone<br/>(16 parallel forward passes)"]
        BB3 --> Feat3["Frame Features: [B, T, 576]"]
        Feat3 --> GRU3["Recurrent GRU Dynamics<br/>[B, 256] (Order-sensitive)"]
        GRU3 --> H3["Linear Head -> Classes"]
    end
```

---

### 3.1 Paradigm 1: Early Fusion (Channel Stacking at Pixel Level)

In **Early Fusion**, the network merges temporal information immediately at the input layer:
1. All $T$ frames are stacked along the channel dimension:

$$
\mathbf{X}_{\text{early}} = \mathbf{X}.\mathrm{view}(B, T \cdot C, H, W) \in \mathbb{R}^{B \times 48 \times 224 \times 224} \quad (\text{for } T=16, C=3)
$$

2. The first convolutional layer (stem) is replaced with a layer whose `in_channels` is $T \cdot C = 48$ rather than $3$.
3. **Channel Weight Inflation**:
   To initialize this new stem layer from an ImageNet-pretrained 2D filter $\mathbf{W}_{\text{2D}} \in \mathbb{R}^{C_{\text{out}} \times 3 \times K_h \times K_w}$, we replicate the 3-channel weights $T$ times across the channel dimension and scale by $1/T$:

$$
\mathbf{W}_{\text{early}}^{(i, c + (t-1)\cdot 3, y, x)} = \frac{1}{T} \mathbf{W}_{\text{2D}}^{(i, c, y, x)}, \quad \forall t \in \{1, \dots, T\}
$$

#### The Static Video Invariance Proof:
Consider a static video where all $T$ frames are identical: $\mathbf{X}(t, c, y, x) = \mathbf{X}_{\text{static}}(c, y, x)$.
The output of the inflated convolution is:

$$
\mathbf{Y}_{\text{early}}(i, y, x) = \sum_{t=1}^T \sum_{c=1}^3 \sum_{u, v} \mathbf{X}_{\text{static}}(c, y+u, x+v) \cdot \left(\frac{1}{T} \mathbf{W}_{\text{2D}}(i, c, u, v)\right)
$$

$$
= \frac{1}{T} \sum_{t=1}^T \left[ \sum_{c=1}^3 \sum_{u, v} \mathbf{X}_{\text{static}}(c, y+u, x+v) \mathbf{W}_{\text{2D}}(i, c, u, v) \right] = \mathbf{Y}_{\text{2D}}(i, y, x)
$$

The response on a static video is **mathematically identical** to standard single-frame 2D inference!

#### Computational & Representational Trade-Offs of Early Fusion:
* **The Speed Advantage**: Notice that after the first convolutional layer, the intermediate tensor shape is $[B, C_1, H_1, W_1]$. The rest of the backbone runs on batch size $B=1$ rather than $B \cdot T = 16$. This makes Early Fusion **$\sim 2.5\times\text{--}3.5\times$ faster** in forward latency than Late Fusion!
* **The Representation Bottleneck**: All temporal interactions are forced to collapse in the very first layer within a small local kernel ($3 \times 3$). The deeper layers have no direct access to inter-frame correspondences or motion trajectories. As empirically shown by Karpathy et al. (2014), Early Fusion often underperforms on complex motions because subtle temporal signals cannot be resolved by low-level spatial filters alone.

---

### 3.2 Paradigm 2: Late Fusion — Temporal Average Pooling

In **Late Fusion (Mean Pooling)**, the 2D CNN backbone processes all $T$ frames independently using the **Batch Flattening Trick**:

$$
\mathbf{X}_{\text{flat}} = \mathbf{X}.\mathrm{view}(B \cdot T, C, H, W) \in \mathbb{R}^{16 \times 3 \times 224 \times 224}
$$

The backbone outputs high-level semantic feature vectors $\mathbf{h}_t \in \mathbb{R}^{576}$ for each frame $t$. We then collapse the temporal dimension via average pooling:

$$
\mathbf{z}_{\text{mean}} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{h}_t \in \mathbb{R}^{D}
$$

* **Inductive Bias**: Permutation-invariant (shuffling video frames produces the exact same classification prediction).
* **Pros**: Zero additional parameters ($932\text{k}$ total), fast, highly robust for static actions (e.g., *playing cello*, *sitting in classroom*).
* **Cons**: Cannot distinguish irreversible directional actions (e.g., *opening a door* vs *closing a door*).

---

### 3.3 Paradigm 3: Late Fusion — Recurrent GRU Temporal Modeling

Instead of order-invariant pooling, we feed the sequence of frame vectors $[\mathbf{h}_1, \dots, \mathbf{h}_T]$ into a Gated Recurrent Unit ([Cho et al., EMNLP 2014](https://arxiv.org/abs/1406.1078); [Donahue et al., CVPR 2015](https://arxiv.org/abs/1411.4389)):

$$
\mathbf{s}_t = \mathrm{GRU}(\mathbf{h}_t, \mathbf{s}_{t-1}), \quad \mathbf{z}_{\text{gru}} = \mathbf{s}_T \in \mathbb{R}^{d_h}
$$

* **Inductive Bias**: Order-sensitive temporal dynamics via hidden state memory.
* **Pros**: Explicitly models sequential transitions and motion directionality.
* **Cons**: Adds recurrent parameters ($3 \times 256 \times (576 + 256 + 2) = 640,512 \approx 640.5\text{k}$ parameters across the 3 GRU gates); higher forward latency ($\approx 65\text{ ms}$ on MPS).

---

### 3.4 Head-to-Head Architectural Comparison

| Dimension | Early Fusion | Late Fusion (Mean) | Late Fusion (GRU) |
| :--- | :--- | :--- | :--- |
| **Temporal Merging Point** | Pixel-level (Stem layer `conv1`) | Embedding-level (Post-backbone) | Embedding-level (Post-backbone) |
| **Stem Input Tensor** | $[B, T \cdot C, H, W] = [1, 48, 224, 224]$ | $[B \cdot T, C, H, W] = [16, 3, 224, 224]$ | $[B \cdot T, C, H, W] = [16, 3, 224, 224]$ |
| **Backbone Forward Passes** | **$1$ forward pass per video clip** | **$T$ forward passes ($16$ frames)** | **$T$ forward passes ($16$ frames)** |
| **Total Model Parameters** | $939,258$ ($+6,480$ stem params) | $932,778$ (ImageNet baseline) | $1,570,090$ ($+640\text{k}$ GRU params) |
| **Forward Latency (Apple MPS)** | **$\approx 14.8\text{ ms}$ (Fastest)** | $\approx 36.5\text{ ms}$ | $\approx 65.8\text{ ms}$ |
| **Order Sensitivity** | Sensitive to channel ordering | Completely permutation-invariant | Order-sensitive recurrent dynamics |
| **Pretrained Weight Bootstrap** | Channel Inflation ($\mathbf{W}_{\text{2D}} / T$) | Native ImageNet weights unchanged | Native ImageNet weights unchanged |

---

## 4. Tensor Dimension Tracing

| Pipeline Stage | Operation / Function | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Video Demuxing** | Container Index Lookup | Disk MP4 | Seek Offsets | Finds I-frame byte positions via PTS |
| **Video Decoding** | PyAV Fast Seek | Disk Bitstream | `(16, 360, 640, 3)` | Decoded as uint8 RGB NumPy array |
| **Preprocessing** | Resizing & Normalization | `16 x [360, 640, 3]` | `[16, 3, 224, 224]` | Scaled to $[0, 1]$ and normalized with ImageNet stats |
| **Video Batching** | `torch.stack().unsqueeze(0)` | `[16, 3, 224, 224]` | `[1, 16, 3, 224, 224]` | Creates 5D tensor: `[B, T, C, H, W]` |
| **Early Fusion Stem** | `Conv2D(48, 16, 3)` | `[1, 48, 224, 224]` | `[1, 16, 112, 112]` | Time collapsed immediately in first layer |
| **Early Backbone** | MobileNetV3 (Single Pass) | `[1, 16, 112, 112]` | `[1, 576]` | Only 1 spatial map processed across all blocks |
| **Late Backbone** | `view(B*T, 3, 224, 224)` | `[16, 3, 224, 224]` | `[16, 576]` | 16 frames processed in parallel |
| **Late Mean Pooling** | `torch.mean(dim=1)` | `[1, 16, 576]` | `[1, 576]` | Time dimension collapsed |
| **Late GRU Head** | `gru(x)[:, -1, :]` | `[1, 16, 576]` | `[1, 256]` | Hidden dimension $d_h = 256$ |
| **Classification Head** | `Linear(D, K)` | `[1, 576]` or `[1, 256]` | `[1, 10]` | Unnormalized class logits |

---

## 5. Execution Playbook

### 1. Comparative Head-to-Head Benchmark Across All Paradigms
```bash
python video-understanding-lab/01_video_io_and_fusion_paradigms.py --video data/person_bicycle.mp4 --fusion all --device mps
```

### 2. Running Early Fusion
```bash
python video-understanding-lab/01_video_io_and_fusion_paradigms.py --video data/person_bicycle.mp4 --fusion early --device mps
```

### 3. Running Late Fusion (Mean Pooling)
```bash
python video-understanding-lab/01_video_io_and_fusion_paradigms.py --video data/person_bicycle.mp4 --fusion mean --device mps
```

### 4. Running Late Fusion (Recurrent GRU)
```bash
python video-understanding-lab/01_video_io_and_fusion_paradigms.py --video data/person_bicycle.mp4 --fusion gru --device mps
```

### 5. Inspecting Detailed Neural Network Architectures
```bash
python video-understanding-lab/01_video_io_and_fusion_paradigms.py --fusion early --print-model
```

### CLI Arguments Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--video` | `str` | `data/person_bicycle.mp4` | Path to input `.mp4` video file |
| `--num_frames` | `int` | `16` | Number of frames uniformly sampled across video |
| `--fusion` | `str` | `mean` | Fusion paradigm: `early`, `mean`, `gru`, or `all` |
| `--print-model` | `flag` | `False` | Print detailed PyTorch neural network layer architecture |
| `--device` | `str` | `auto` | Target device (`auto`, `mps`, `cuda`, `cpu`) |

---

## 6. Expected Terminal Output & Interpretation

Running the comparative benchmark (`--fusion all --device mps`) outputs:

```text
=================================================================
 PyTorch Version : 2.X.X
 Active Device   : MPS
 Backend         : Apple Silicon Metal Performance Shaders (MPS)
 Note            : Leveraging Unified Memory Architecture (UMA)
=================================================================

--- [STEP 1] Industrial I/O Decoding Benchmark ---
Target Video Clip : data/person_bicycle.mp4 (Sample 16 frames)
  1. OpenCV (Sequential Loop)             : 226.38 ms
  2. PyAV Naive (Single-threaded Loop)    : 367.15 ms
  3. PyAV Fast Seek (Industrial Standard) :  61.04 ms
==> PyAV with Keyframe Seeking is 3.7x faster than OpenCV!

--- [STEP 2] Constructing the 5D Video Tensor ---
Input Video Tensor Shape: (1, 16, 3, 224, 224) (Batch, Time, Channels, Height, Width)
Tensor VRAM Footprint on MPS: 9408.00 KB

--- [STEP 3] Evaluating Fusion Paradigms (ALL) ---

========================================================================================
 HEAD-TO-HEAD COMPARATIVE BENCHMARK: SPATIO-TEMPORAL FUSION PARADIGMS (MPS)
========================================================================================
Fusion Strategy    | Stem Input         | Backbone Passes | Params      | Latency (MPS) 
----------------------------------------------------------------------------------------
Early Fusion       | [1, 48, 224, 224]  | 1               | 939,258     |  14.76 ms
Late Fusion (Mean) | [16, 3, 224, 224]  | 16              | 932,778     |  36.53 ms
Late Fusion (GRU)  | [16, 3, 224, 224]  | 16              | 1,570,090   |  65.83 ms
========================================================================================

Key Pedagogical Takeaways for Students:
  1. Early Fusion is ~2.5x to ~3.5x faster in forward pass latency because the 2D CNN backbone
     processes only 1 spatial map ([1, 48, 224, 224]) rather than 16 frames in parallel.
  2. However, Early Fusion restricts temporal interactions to local 3x3 receptive fields
     in the stem layer, whereas Late Fusion allows the backbone to extract high-level semantic
     features before temporal aggregation.

[SUCCESS] Session 1 demo completed.
```

---

## 7. Self-Study Exercises & Experiments

### Exercise 1: Temporal Invariance Verification
1. Open [`01_video_io_and_fusion_paradigms.py`](../01_video_io_and_fusion_paradigms.py) in your editor.
2. Invert the temporal order of frames before model inference:
   ```python
   video_tensor = torch.flip(video_tensor, dims=[1])
   ```
3. Run `--fusion early`, `--fusion mean`, and `--fusion gru`.
4. **Questions**:
   - Why does Late Fusion (`mean`) produce the exact same logits when time is reversed?
   - Does Early Fusion produce different logits when channels are reversed? Why?
   - How does GRU react to reversing time?

### Exercise 2: Scaling Number of Frames
Test the script with `--num_frames 8`, `--num_frames 16`, and `--num_frames 32`:
* How does the input stem dimension change for Early Fusion ($3 \times 8 = 24$, $3 \times 16 = 48$, $3 \times 32 = 96$)?
* How does forward pass latency scale for Early Fusion vs Late Fusion?

---

## 8. Troubleshooting FAQ

* **Error: `FileNotFoundError: Could not find demo video`**:
  * Run `python video-understanding-lab/download_samples.py` from the repository root to download the sample video clip into `data/`.
* **Warning: `Class AVFFrameReceiver is implemented in both...`**:
  * This is an informational macOS dynamic linker warning caused when both OpenCV and PyAV bundle FFmpeg dylibs. It does not affect execution or numerical correctness.
* **MPS Out-of-Memory or Unsupported Op Warning**:
  * If running on an older macOS version without full MPS support, pass `--device cpu` to verify functionality.

---

## 📚 References & Canonical Literature

1. **Large-Scale Video Classification & Fusion Paradigms:**
   - Karpathy, A., Toderici, G., Shetty, S., Leung, T., Sukthankar, R., & Fei-Fei, L. (2014). "Large-scale Video Classification with Convolutional Neural Networks". *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2014)*, 1725–1732. [Paper Link](https://cs.stanford.edu/people/karpathy/deepvideo/)

2. **Video Compression & Codec Standards:**
   - Wiegand, T., Sullivan, G. J., Bjøntegaard, G., & Luthra, A. (2003). "Overview of the H.264/AVC Video Coding Standard". *IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)*, 13(7), 560–576. [IEEE Xplore](https://ieeexplore.ieee.org/document/1218189)
   - Sullivan, G. J., Ohm, J. R., Han, W. J., & Wiegand, T. (2012). "Overview of the High Efficiency Video Coding (HEVC) Standard". *IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)*, 22(12), 1649–1668. [IEEE Xplore](https://ieeexplore.ieee.org/document/6316136)

3. **2D Deep Backbone Architectures & Transfer Learning:**
   - Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). "ImageNet: A Large-Scale Hierarchical Image Database". *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2009)*. [IEEE Xplore](https://ieeexplore.ieee.org/document/5206848)
   - He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition". *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016)*. [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
   - Howard, A., et al. (2019). "Searching for MobileNetV3". *IEEE/CVF International Conference on Computer Vision (ICCV 2019)*. [https://arxiv.org/abs/1905.02244](https://arxiv.org/abs/1905.02244)

4. **Recurrent Spatio-Temporal Modeling:**
   - Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation". *Conference on Empirical Methods in Natural Language Processing (EMNLP 2014)*. [https://arxiv.org/abs/1406.1078](https://arxiv.org/abs/1406.1078)
   - Donahue, J., Anne Hendricks, L., Guadarrama, S., Rohrbach, M., Venugopalan, S., Saenko, K., & Darrell, T. (2015). "Long-term Recurrent Convolutional Networks for Visual Recognition and Description". *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2015)*. [https://arxiv.org/abs/1411.4389](https://arxiv.org/abs/1411.4389)
