"""
Session 1 — Lab A: Quantization (FP32 vs. Dynamic PTQ vs. Static PTQ vs. QAT)
Uses a pretrained torchvision ResNet-18 adapted to CIFAR-10.

Demonstrates:
1. Fine-tuning a pretrained ResNet-18 on CIFAR-10 (or loading cached checkpoint).
2. Exporting FP32 ONNX and applying ONNX Runtime Dynamic INT8 PTQ.
3. Applying ONNX Runtime Static INT8 PTQ with a real 100-image CalibrationDataReader.
4. Running PyTorch Quantization-Aware Training (QAT) with FakeQuantize + STE.
5. Comparing Size (MB), Top-1 Accuracy (%), and FPS across all 4 precision modes.
"""

import argparse
import os
import platform
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
import torch
import torch.nn as nn
from torchvision import models

from utils import (
    DEFAULT_CHECKPOINT_DIR,
    benchmark_callable,
    evaluate_classifier,
    export_to_onnx,
    get_cifar10_loaders,
    get_device,
    get_file_size_mb,
)


class QuantizableResNet18(nn.Module):
    """Uses torchvision.models.quantization.resnet18 (FloatFunctional skip-adds) for PyTorch QAT."""

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.net = models.quantization.resnet18(weights=weights, quantize=False)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CifarCalibrationReader(CalibrationDataReader):
    """Feeds real CIFAR-10 validation images into ONNX Runtime Static PTQ."""

    def __init__(self, loader, input_name: str, max_samples: int = 100):
        self.samples: list[dict[str, np.ndarray]] = []
        count = 0
        for images, _ in loader:
            for i in range(images.size(0)):
                if count >= max_samples:
                    break
                arr = images[i : i + 1].numpy().astype(np.float32)
                self.samples.append({input_name: arr})
                count += 1
            if count >= max_samples:
                break
        self._iter = iter(self.samples)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iter, None)


def train_epochs(
    model: nn.Module,
    loader,
    device: torch.device,
    epochs: int,
    lr: float,
    log_interval: int = 20,
) -> None:
    """Trains `model` for `epochs` on `device`, logging mid-epoch batch progress."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    num_batches = len(loader)
    for ep in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        running_loss = 0.0
        seen = 0
        for step, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            bs = labels.size(0)
            running_loss += loss.item() * bs
            seen += bs
            if log_interval > 0 and (step % log_interval == 0 or step == num_batches):
                elapsed = time.perf_counter() - ep_t0
                print(
                    f"  [Train] Epoch {ep}/{epochs} [{step}/{num_batches}]"
                    f" — loss: {running_loss / seen:.4f} ({elapsed:.1f}s)",
                    flush=True,
                )


def eval_onnx(onnx_path: str, val_loader) -> tuple[float, dict[str, float]]:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    correct, total = 0, 0
    for images, labels in val_loader:
        for i in range(images.size(0)):
            out = sess.run(None, {input_name: images[i : i + 1].numpy().astype(np.float32)})[0]
            if int(np.argmax(out, axis=1)[0]) == int(labels[i].item()):
                correct += 1
            total += 1
    acc = 100.0 * correct / max(total, 1)
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    stats = benchmark_callable(lambda: sess.run(None, {input_name: dummy}), warmup=10, iterations=60)
    return acc, stats


def main():
    parser = argparse.ArgumentParser(description="IF5152 Quantization: PTQ (Dynamic/Static) & QAT")
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, or cpu")
    parser.add_argument("--quick", action="store_true", help="Use CIFAR-10 subset for fast classroom demo")
    parser.add_argument("--epochs", type=int, default=2, help="Fine-tuning epochs for CIFAR-10 head")
    args = parser.parse_args()

    os.makedirs(DEFAULT_CHECKPOINT_DIR, exist_ok=True)
    device = get_device(args.device)
    print(f"[Setup] Using device: {device}")

    train_sub = 1000 if args.quick else 5000
    val_sub = 200 if args.quick else 1000
    train_loader, val_loader = get_cifar10_loaders(
        batch_size=32, img_size=224, train_subset=train_sub, val_subset=val_sub
    )

    # 1. Load or fine-tune pretrained ResNet-18 on CIFAR-10
    ckpt_fp32 = os.path.join(DEFAULT_CHECKPOINT_DIR, "resnet18_cifar10_fp32.pth")
    model = QuantizableResNet18(num_classes=10, pretrained=True).to(device)
    if os.path.exists(ckpt_fp32):
        print(f"[1/4] Loading cached FP32 checkpoint: {ckpt_fp32}")
        model.load_state_dict(torch.load(ckpt_fp32, map_location=device))
    else:
        print(f"[1/4] Fine-tuning pretrained ResNet-18 on CIFAR-10 ({train_sub} imgs, {args.epochs} epochs)...")
        train_epochs(model, train_loader, device, epochs=args.epochs, lr=3e-4)
        torch.save(model.state_dict(), ckpt_fp32)

    fp32_acc = evaluate_classifier(model, val_loader, device)
    print(f"  FP32 PyTorch Val Accuracy: {fp32_acc:.2f}%")

    # 2. Export FP32 ONNX
    model_cpu = model.to("cpu").eval()
    onnx_fp32 = os.path.join(DEFAULT_CHECKPOINT_DIR, "resnet18_fp32.onnx")
    dummy = torch.randn(1, 3, 224, 224)
    export_to_onnx(
        model_cpu, dummy, onnx_fp32,
        opset_version=17,
        input_names=["input"], output_names=["logits"],
    )
    onnx.checker.check_model(onnx.load(onnx_fp32))
    acc_fp32_ort, stats_fp32 = eval_onnx(onnx_fp32, val_loader)

    # 3. Dynamic INT8 PTQ (ONNX Runtime)
    print("[2/4] Running Dynamic INT8 Post-Training Quantization...")
    onnx_dyn = os.path.join(DEFAULT_CHECKPOINT_DIR, "resnet18_int8_dynamic.onnx")
    quantize_dynamic(model_input=onnx_fp32, model_output=onnx_dyn, weight_type=QuantType.QUInt8)
    acc_dyn, stats_dyn = eval_onnx(onnx_dyn, val_loader)

    # 4. Static INT8 PTQ (ONNX Runtime with real CIFAR-10 calibration)
    print("[3/4] Running Static INT8 PTQ (100-image CIFAR-10 calibration)...")
    onnx_stat = os.path.join(DEFAULT_CHECKPOINT_DIR, "resnet18_int8_static.onnx")
    cal_reader = CifarCalibrationReader(val_loader, input_name="input", max_samples=100)
    quantize_static(
        model_input=onnx_fp32,
        model_output=onnx_stat,
        calibration_data_reader=cal_reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    acc_stat, stats_stat = eval_onnx(onnx_stat, val_loader)

    # 5. PyTorch Quantization-Aware Training (QAT)
    print("[4/4] Running PyTorch Quantization-Aware Training (QAT)...")
    backend = "qnnpack" if platform.machine().lower() in ("arm64", "aarch64") else "fbgemm"
    torch.backends.quantized.engine = backend
    qat_model = QuantizableResNet18(num_classes=10, pretrained=False)
    qat_model.load_state_dict(torch.load(ckpt_fp32, map_location="cpu"))
    qat_model.train()
    qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    torch.ao.quantization.prepare_qat(qat_model, inplace=True)
    # 1 quick QAT epoch on CPU (FakeQuantize + STE)
    train_epochs(qat_model, train_loader, torch.device("cpu"), epochs=1, lr=5e-5)
    qat_model.eval()
    qat_int8 = torch.ao.quantization.convert(qat_model, inplace=False)
    acc_qat = evaluate_classifier(qat_int8, val_loader, torch.device("cpu"))
    qat_ckpt = os.path.join(DEFAULT_CHECKPOINT_DIR, "resnet18_qat_int8.pth")
    torch.save(qat_int8.state_dict(), qat_ckpt)
    stats_qat = benchmark_callable(lambda: qat_int8(dummy), warmup=10, iterations=60)

    # Summary Table
    print("\n" + "=" * 78)
    print(f"{'Method':<26} {'Size (MB)':>10} {'Val Acc (%)':>12} {'Mean (ms)':>11} {'FPS':>10}")
    print("-" * 78)
    rows = [
        ("FP32 Baseline (ONNX)", get_file_size_mb(onnx_fp32), acc_fp32_ort, stats_fp32),
        ("Dynamic INT8 PTQ (ORT)", get_file_size_mb(onnx_dyn), acc_dyn, stats_dyn),
        ("Static INT8 PTQ (ORT)", get_file_size_mb(onnx_stat), acc_stat, stats_stat),
        (f"PyTorch QAT ({backend})", get_file_size_mb(qat_ckpt), acc_qat, stats_qat),
    ]
    for name, mb, acc, st in rows:
        print(f"{name:<26} {mb:>10.2f} {acc:>11.2f}% {st['mean_ms']:>11.2f} {st['fps']:>10.1f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
