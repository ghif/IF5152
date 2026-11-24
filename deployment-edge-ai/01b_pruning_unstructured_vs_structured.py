"""
Session 1 — Lab B: Unstructured vs. Structured Channel Pruning
Uses a pretrained torchvision VGG-11-BN adapted to CIFAR-10.

Demonstrates:
1. Unstructured L1 weight pruning (70% sparsity): high sparsity, zero FPS/MB gain.
2. Structured Channel Pruning (50% channels removed via BatchNorm gamma / L1 norm):
   physically rebuilds narrower Conv2d + BatchNorm2d layers, fine-tunes briefly,
   and achieves real Parameter, Size (MB), and wall-clock FPS improvements.
"""

import argparse
import copy
import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
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


def build_vgg11_cifar10(pretrained: bool = True) -> nn.Module:
    weights = models.VGG11_BN_Weights.DEFAULT if pretrained else None
    model = models.vgg11_bn(weights=weights)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 10),
    )
    return model


def train_epochs(model: nn.Module, loader, device: torch.device, epochs: int, lr: float) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for ep in range(1, epochs + 1):
        loss_sum = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
        print(f"  [Fine-tune] Epoch {ep}/{epochs} — loss: {loss_sum / len(loader.dataset):.4f}")


def apply_unstructured_pruning(model: nn.Module, amount: float = 0.70) -> nn.Module:
    """Applies L1 unstructured pruning to all Conv2d layers and removes reparam wrappers."""
    pruned = copy.deepcopy(model)
    for m in pruned.modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name="weight", amount=amount)
            prune.remove(m, "weight")
    return pruned


def measure_weight_sparsity(model: nn.Module) -> float:
    zeros, total = 0, 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            zeros += int((m.weight == 0).sum().item())
            total += m.weight.numel()
    return 100.0 * zeros / max(total, 1)


def apply_structured_channel_pruning(model: nn.Module, keep_ratio: float = 0.5) -> nn.Module:
    """
    Physically prunes Conv2d output channels (and coupled BatchNorm2d + next Conv2d input channels)
    ranked by BatchNorm |gamma| scaling factor (or Conv L1 filter norm).
    """
    pruned = copy.deepcopy(model).to("cpu")
    features = list(pruned.features.children())
    new_features: list[nn.Module] = []

    in_keep_idx = torch.arange(3)  # RGB input channels
    i = 0
    while i < len(features):
        layer = features[i]
        if isinstance(layer, nn.Conv2d):
            bn_layer = features[i + 1] if (i + 1 < len(features) and isinstance(features[i + 1], nn.BatchNorm2d)) else None
            out_ch = layer.out_channels
            keep_k = max(8, int(out_ch * keep_ratio))

            # Rank output channels by BatchNorm |gamma| if available, else Conv L1 norm
            if bn_layer is not None:
                scores = bn_layer.weight.data.abs()
            else:
                scores = layer.weight.data.abs().sum(dim=(1, 2, 3))

            out_keep_idx = torch.topk(scores, k=keep_k, largest=True).indices.sort().values

            new_conv = nn.Conv2d(
                in_channels=len(in_keep_idx),
                out_channels=keep_k,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                bias=(layer.bias is not None),
            )
            with torch.no_grad():
                w_sub = layer.weight.data[out_keep_idx][:, in_keep_idx, :, :]
                new_conv.weight.copy_(w_sub)
                if layer.bias is not None:
                    new_conv.bias.copy_(layer.bias.data[out_keep_idx])
            new_features.append(new_conv)

            if bn_layer is not None:
                new_bn = nn.BatchNorm2d(keep_k)
                with torch.no_grad():
                    new_bn.weight.copy_(bn_layer.weight.data[out_keep_idx])
                    new_bn.bias.copy_(bn_layer.bias.data[out_keep_idx])
                    new_bn.running_mean.copy_(bn_layer.running_mean[out_keep_idx])
                    new_bn.running_var.copy_(bn_layer.running_var[out_keep_idx])
                new_features.append(new_bn)
                i += 1  # consumed BN

            in_keep_idx = out_keep_idx
        else:
            new_features.append(layer)
        i += 1

    pruned.features = nn.Sequential(*new_features)

    # Adapt first Linear layer in classifier to match pruned final Conv2d channels (len(in_keep_idx) * 7 * 7)
    old_fc0 = pruned.classifier[0]
    assert isinstance(old_fc0, nn.Linear)
    new_in_features = len(in_keep_idx) * 7 * 7
    new_fc0 = nn.Linear(new_in_features, old_fc0.out_features)
    with torch.no_grad():
        w_reshaped = old_fc0.weight.data.view(old_fc0.out_features, -1, 7, 7)
        w_pruned = w_reshaped[:, in_keep_idx, :, :].reshape(old_fc0.out_features, new_in_features)
        new_fc0.weight.copy_(w_pruned)
        new_fc0.bias.copy_(old_fc0.bias.data)
    pruned.classifier[0] = new_fc0
    return pruned


def main():
    parser = argparse.ArgumentParser(description="IF5152 Pruning: Unstructured vs. Structured Channel")
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, or cpu")
    parser.add_argument("--quick", action="store_true", help="Use CIFAR-10 subset for fast classroom demo")
    args = parser.parse_args()

    os.makedirs(DEFAULT_CHECKPOINT_DIR, exist_ok=True)
    device = get_device(args.device)
    print(f"[Setup] Using device: {device}")

    train_sub = 800 if args.quick else 4000
    val_sub = 200 if args.quick else 1000
    train_loader, val_loader = get_cifar10_loaders(
        batch_size=32, img_size=224, train_subset=train_sub, val_subset=val_sub
    )

    ckpt_path = os.path.join(DEFAULT_CHECKPOINT_DIR, "vgg11bn_cifar10.pth")
    base_model = build_vgg11_cifar10(pretrained=True).to(device)
    if os.path.exists(ckpt_path):
        print(f"[1/3] Loading cached VGG-11-BN checkpoint: {ckpt_path}")
        base_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print("[1/3] Fine-tuning pretrained VGG-11-BN on CIFAR-10...")
        train_epochs(base_model, train_loader, device, epochs=2, lr=2e-4)
        torch.save(base_model.state_dict(), ckpt_path)

    dummy = torch.randn(1, 3, 224, 224, device=device)
    base_model.eval()
    acc_base = evaluate_classifier(base_model, val_loader, device)
    st_base = benchmark_callable(lambda: base_model(dummy), warmup=10, iterations=50, sync_fn=lambda: sync_device(device))

    # 2. Unstructured L1 Pruning (70%)
    print("[2/3] Applying Unstructured L1 Pruning (70% zeroed weights)...")
    unstruct = apply_unstructured_pruning(base_model, amount=0.70).to(device).eval()
    sparsity_unstruct = measure_weight_sparsity(unstruct)
    acc_unstruct = evaluate_classifier(unstruct, val_loader, device)
    st_unstruct = benchmark_callable(lambda: unstruct(dummy), warmup=10, iterations=50, sync_fn=lambda: sync_device(device))

    # 3. Structured Channel Pruning (50% channels removed physically)
    print("[3/3] Applying Structured Channel Pruning (50% channels removed) + 1-epoch fine-tune...")
    struct = apply_structured_channel_pruning(base_model, keep_ratio=0.50).to(device)
    train_epochs(struct, train_loader, device, epochs=1, lr=1e-4)
    struct.eval()
    acc_struct = evaluate_classifier(struct, val_loader, device)
    st_struct = benchmark_callable(lambda: struct(dummy), warmup=10, iterations=50, sync_fn=lambda: sync_device(device))
    torch.save(struct.state_dict(), os.path.join(DEFAULT_CHECKPOINT_DIR, "vgg11bn_structured_pruned.pth"))

    print("\n" + "=" * 86)
    print(f"{'Model Variant':<30} {'Params (M)':>10} {'Size (MB)':>10} {'Sparsity':>9} {'Acc (%)':>9} {'FPS':>9}")
    print("-" * 86)
    rows = [
        ("Baseline VGG-11-BN", base_model, 0.0, acc_base, st_base),
        ("Unstructured L1 (70% mask)", unstruct, sparsity_unstruct, acc_unstruct, st_unstruct),
        ("Structured Channel (50% cut)", struct, 0.0, acc_struct, st_struct),
    ]
    for name, m, sp, acc, st in rows:
        params_m = count_parameters(m) / 1e6
        mb = get_state_dict_size_mb(m)
        print(f"{name:<30} {params_m:>10.2f} {mb:>10.2f} {sp:>8.1f}% {acc:>8.2f}% {st['fps']:>9.1f}")
    print("=" * 86)
    print("Key Takeaway: Unstructured masking keeps tensor shapes identical (0% speedup);")
    print("              Structured channel pruning physically shrinks layers for real FPS & MB gains.")


if __name__ == "__main__":
    main()
