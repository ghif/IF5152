"""
Session 1 — Lab C: Knowledge Distillation (Teacher -> Student)
Uses a pretrained torchvision ResNet-34 Teacher distilling into a compact
MobileNetV3-Small Student on CIFAR-10.

Demonstrates:
1. Fine-tuning (or loading cached) ResNet-34 Teacher on CIFAR-10.
2. Training MobileNetV3-Small Student with hard labels only (baseline).
3. Training MobileNetV3-Small Student with Hinton Knowledge Distillation:
   L_KD = (1 - alpha) * CE(y, z_S) + alpha * T^2 * KL(softmax(z_T/T) || softmax(z_S/T))
4. Comparing Size (MB), Val Accuracy (%), and FPS.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from utils import (
    DEFAULT_CHECKPOINT_DIR,
    benchmark_callable,
    count_parameters,
    evaluate_classifier,
    get_cifar10_loaders,
    get_device,
    get_state_dict_size_mb,
    sync_device,
)


def build_resnet34_teacher(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
    model = models.resnet34(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_mobilenet_student(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feat, num_classes)
    return model


def train_teacher(model: nn.Module, loader, device: torch.device, epochs: int) -> None:
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    model.train()
    for ep in range(1, epochs + 1):
        loss_sum = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * labels.size(0)
        print(f"  [Teacher] Epoch {ep}/{epochs} — loss: {loss_sum / len(loader.dataset):.4f}")


def train_student(
    student: nn.Module,
    loader,
    device: torch.device,
    epochs: int,
    teacher: nn.Module | None = None,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> None:
    opt = torch.optim.AdamW(student.parameters(), lr=5e-4, weight_decay=1e-4)
    if teacher is not None:
        teacher.eval()
    student.train()
    tag = "Distilled Student" if teacher is not None else "Baseline Student"

    for ep in range(1, epochs + 1):
        loss_sum = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            z_s = student(images)
            hard_loss = F.cross_entropy(z_s, labels)

            if teacher is not None:
                with torch.no_grad():
                    z_t = teacher(images)
                p_t = F.softmax(z_t / temperature, dim=1)
                log_p_s = F.log_softmax(z_s / temperature, dim=1)
                kl_loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature ** 2)
                loss = (1.0 - alpha) * hard_loss + alpha * kl_loss
            else:
                loss = hard_loss

            loss.backward()
            opt.step()
            loss_sum += loss.item() * labels.size(0)
        print(f"  [{tag}] Epoch {ep}/{epochs} — loss: {loss_sum / len(loader.dataset):.4f}")


def main():
    parser = argparse.ArgumentParser(description="IF5152 Knowledge Distillation: ResNet-34 -> MobileNetV3-Small")
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, or cpu")
    parser.add_argument("--quick", action="store_true", help="Use CIFAR-10 subset for fast classroom demo")
    parser.add_argument("--temperature", type=float, default=4.0, help="Softmax temperature T")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight on KL distillation loss")
    args = parser.parse_args()

    os.makedirs(DEFAULT_CHECKPOINT_DIR, exist_ok=True)
    device = get_device(args.device)
    print(f"[Setup] Using device: {device} | T={args.temperature}, alpha={args.alpha}")

    train_sub = 1000 if args.quick else 5000
    val_sub = 200 if args.quick else 1000
    epochs = 2 if args.quick else 5
    train_loader, val_loader = get_cifar10_loaders(
        batch_size=32, img_size=224, train_subset=train_sub, val_subset=val_sub
    )

    # 1. Teacher (ResNet-34)
    teacher_ckpt = os.path.join(DEFAULT_CHECKPOINT_DIR, "resnet34_teacher_cifar10.pth")
    teacher = build_resnet34_teacher(num_classes=10, pretrained=True).to(device)
    if os.path.exists(teacher_ckpt):
        print(f"[1/3] Loading cached Teacher checkpoint: {teacher_ckpt}")
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    else:
        print(f"[1/3] Fine-tuning ResNet-34 Teacher on CIFAR-10 ({epochs} epochs)...")
        train_teacher(teacher, train_loader, device, epochs=epochs)
        torch.save(teacher.state_dict(), teacher_ckpt)

    teacher.eval()
    acc_teacher = evaluate_classifier(teacher, val_loader, device)

    # 2. Baseline Student (MobileNetV3-Small, hard labels only)
    print(f"[2/3] Training Baseline Student (MobileNetV3-Small, hard labels only, {epochs} epochs)...")
    student_base = build_mobilenet_student(num_classes=10, pretrained=False).to(device)
    train_student(student_base, train_loader, device, epochs=epochs, teacher=None)
    student_base.eval()
    acc_base = evaluate_classifier(student_base, val_loader, device)

    # 3. Distilled Student (MobileNetV3-Small, Teacher soft targets)
    print(f"[3/3] Training Distilled Student (MobileNetV3-Small, KD from ResNet-34, {epochs} epochs)...")
    student_kd = build_mobilenet_student(num_classes=10, pretrained=False).to(device)
    train_student(
        student_kd, train_loader, device, epochs=epochs,
        teacher=teacher, temperature=args.temperature, alpha=args.alpha,
    )
    student_kd.eval()
    acc_kd = evaluate_classifier(student_kd, val_loader, device)
    kd_ckpt = os.path.join(DEFAULT_CHECKPOINT_DIR, "mobilenetv3_distilled.pth")
    torch.save(student_kd.state_dict(), kd_ckpt)

    dummy = torch.randn(1, 3, 224, 224, device=device)
    st_teacher = benchmark_callable(lambda: teacher(dummy), warmup=10, iterations=50, sync_fn=lambda: sync_device(device))
    st_kd = benchmark_callable(lambda: student_kd(dummy), warmup=10, iterations=50, sync_fn=lambda: sync_device(device))

    print("\n" + "=" * 82)
    print(f"{'Model':<32} {'Params (M)':>10} {'Size (MB)':>10} {'Acc (%)':>10} {'FPS':>10}")
    print("-" * 82)
    print(f"{'Teacher (ResNet-34)':<32} {count_parameters(teacher)/1e6:>10.2f} {get_state_dict_size_mb(teacher):>10.2f} {acc_teacher:>9.2f}% {st_teacher['fps']:>10.1f}")
    print(f"{'Student Baseline (Hard Labels)':<32} {count_parameters(student_base)/1e6:>10.2f} {get_state_dict_size_mb(student_base):>10.2f} {acc_base:>9.2f}% {st_kd['fps']:>10.1f}")
    print(f"{'Student Distilled (KD T=4.0)':<32} {count_parameters(student_kd)/1e6:>10.2f} {get_state_dict_size_mb(student_kd):>10.2f} {acc_kd:>9.2f}% {st_kd['fps']:>10.1f}")
    print("=" * 82)
    print(f"Saved distilled student checkpoint -> {kd_ckpt}")


if __name__ == "__main__":
    main()
