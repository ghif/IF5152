# Deployment Model & Edge AI Laboratory
### IF5152: Advanced Computer Vision & Deep Learning — Institut Teknologi Bandung

This directory contains the hands-on Python laboratory modules for **Deployment Model and Edge AI (Sessions 1 & 2)**.


---

## Practical Python Modules

| Script | Session | Topic & Pretrained Model / Use Case |
| :--- | :--- | :--- |
| [`01a_quantization_ptq_and_qat.py`](./01a_quantization_ptq_and_qat.py) | Session 1 | **Quantization:** Pretrained `ResNet-18` on CIFAR-10 — FP32 vs. Dynamic INT8 PTQ vs. Static INT8 PTQ (100-img calibration) vs. PyTorch QAT |
| [`01b_pruning_unstructured_vs_structured.py`](./01b_pruning_unstructured_vs_structured.py) | Session 1 | **Pruning:** Pretrained `VGG-11-BN` on CIFAR-10 — Unstructured L1 masking (70%) vs. Physical Structured Channel Pruning (50%) |
| [`01c_knowledge_distillation.py`](./01c_knowledge_distillation.py) | Session 1 | **Knowledge Distillation:** Pretrained `ResNet-34` Teacher → `MobileNetV3-Small` Student on CIFAR-10 ($T=4.0, \alpha=0.7$) |
| [`02_export_and_verify_onnx.py`](./02_export_and_verify_onnx.py) | Session 2 | **Lab 1A:** Export PyTorch (`mobilenet_v3_small`, `resnet18`, or `ssdlite320_mobilenet_v3_large`) → `.onnx`, validate & assert numerical equivalence |
| [`03_benchmark_onnx_and_trt.py`](./03_benchmark_onnx_and_trt.py) | Session 2 | **Lab 1B:** Benchmark PyTorch Eager vs. ONNX Runtime (Size MB, Mean ms, p95 ms, FPS) + `trtexec` TensorRT command |
| [`04_fastapi_server.py`](./04_fastapi_server.py) | Session 2 | **Lab 2:** Async FastAPI `/health` & `/predict` inference server with lifespan singleton ONNX Runtime session |
| [`05_test_api_client.py`](./05_test_api_client.py) | Session 2 | **Lab 2 Test:** Automated client test (`TestClient` or live `--url http://localhost:8000`) |

---

## Quickstart

```bash
cd IF5152/deployment-edge-ai
pip install -r requirements.txt

# Session 1: Model Optimization (use --quick for fast classroom demos)
python3 01a_quantization_ptq_and_qat.py --quick
python3 01b_pruning_unstructured_vs_structured.py --quick
python3 01c_knowledge_distillation.py --quick

# Session 2: ONNX Export, Benchmarking & FastAPI Serving
python3 02_export_and_verify_onnx.py --model mobilenet_v3_small
python3 03_benchmark_onnx_and_trt.py --model mobilenet_v3_small
python3 05_test_api_client.py

# Optional: Run live FastAPI server & open Swagger UI at http://localhost:8000/docs
uvicorn 04_fastapi_server:app --host 0.0.0.0 --port 8000
```

## NVIDIA Edge / TensorRT CLI Reference
On an NVIDIA GPU (Colab, workstation, or Jetson Orin/Nano):
```bash
trtexec --onnx=checkpoints/mobilenet_v3_small.onnx \
        --saveEngine=checkpoints/mobilenet_v3_small_fp16.engine \
        --fp16 --memPoolSize=workspace:2048M
```
