"""
Session 2 — Lab 1B: Benchmark PyTorch Eager vs. ONNX Runtime & Print TensorRT CLI.
Outputs Size (MB), Mean Latency (ms), p95 Latency (ms), and FPS for the
Week 14 Class Master Google Sheet.
"""

import argparse
import os
import numpy as np
import onnxruntime as ort
import torch

from utils import (
    DEFAULT_CHECKPOINT_DIR,
    benchmark_callable,
    get_device,
    get_file_size_mb,
    get_state_dict_size_mb,
    sync_device,
)


def main():
    parser = argparse.ArgumentParser(description="IF5152 Lab 1B: PyTorch vs ONNX Runtime Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v3_small",
        choices=["mobilenet_v3_small", "resnet18", "ssdlite320_mobilenet_v3_large"],
    )
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, or cpu for PyTorch baseline")
    parser.add_argument("--iterations", type=int, default=100, help="Timed iterations (default: 100)")
    args = parser.parse_args()

    onnx_path = os.path.join(DEFAULT_CHECKPOINT_DIR, f"{args.model}.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"Missing {onnx_path}. Run: python3 02_export_and_verify_onnx.py --model {args.model}"
        )

    from importlib import import_module
    export_mod = import_module("02_export_and_verify_onnx")
    pt_model, input_shape = export_mod.load_pretrained_model(args.model)

    device = get_device(args.device)
    pt_model = pt_model.to(device).eval()
    pt_input = torch.randn(*input_shape, device=device)

    print(f"[1/2] Benchmarking PyTorch Eager ({device.type}) over {args.iterations} runs...")
    with torch.no_grad():
        st_pt = benchmark_callable(
            lambda: pt_model(pt_input),
            warmup=15,
            iterations=args.iterations,
            sync_fn=lambda: sync_device(device),
        )
    pt_mb = get_state_dict_size_mb(pt_model)

    # Benchmark ONNX Runtime on available providers
    avail_eps = ort.get_available_providers()
    ort_input = np.random.randn(*input_shape).astype(np.float32)
    onnx_mb = get_file_size_mb(onnx_path)

    results = [(f"PyTorch Eager ({device.type})", pt_mb, st_pt)]
    for ep in avail_eps:
        if ep in ("AzureExecutionProvider",):
            continue
        try:
            sess = ort.InferenceSession(onnx_path, providers=[ep])
            in_name = sess.get_inputs()[0].name
            print(f"[2/2] Benchmarking ONNX Runtime ({ep}) over {args.iterations} runs...")
            st_ort = benchmark_callable(
                lambda s=sess, n=in_name: s.run(None, {n: ort_input}),
                warmup=15,
                iterations=args.iterations,
            )
            results.append((f"ONNX ({ep.replace('ExecutionProvider', '')})", onnx_mb, st_ort))
        except Exception as exc:
            print(f"  Skipping {ep}: {exc}")

    print("\n" + "=" * 78)
    print("WEEK 14 CLASS MASTER SHEET — BENCHMARK SUMMARY")
    print("=" * 78)
    print(f"{'Runtime / Format':<32} {'Size (MB)':>10} {'Mean (ms)':>11} {'p95 (ms)':>10} {'FPS':>10}")
    print("-" * 78)
    for label, mb, st in results:
        print(f"{label:<32} {mb:>10.2f} {st['mean_ms']:>11.2f} {st['p95_ms']:>10.2f} {st['fps']:>10.1f}")
    print("=" * 78)

    print("\n[NVIDIA Edge / TensorRT Compilation Command]")
    print(
        f"  trtexec --onnx={onnx_path} "
        f"--saveEngine={onnx_path.replace('.onnx', '_fp16.engine')} "
        f"--fp16 --memPoolSize=workspace:2048M"
    )


if __name__ == "__main__":
    main()
