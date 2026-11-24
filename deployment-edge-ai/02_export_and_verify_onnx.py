"""
Session 2 — Lab 1A: Export PyTorch CV Model to ONNX & Verify Numerical Equivalence.
Supports pretrained torchvision classifiers (mobilenet_v3_small, resnet18) and
edge object detector (ssdlite320_mobilenet_v3_large).
"""

import argparse
import os
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision import models

from utils import DEFAULT_CHECKPOINT_DIR, export_to_onnx, get_file_size_mb


def load_pretrained_model(name: str, weights_path: str | None = None) -> tuple[torch.nn.Module, tuple[int, ...]]:
    if name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        shape = (1, 3, 224, 224)
    elif name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        shape = (1, 3, 224, 224)
    elif name == "ssdlite320_mobilenet_v3_large":
        model = models.detection.ssdlite320_mobilenet_v3_large(
            weights=models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        shape = (1, 3, 320, 320)
    else:
        raise ValueError(f"Unsupported model: {name}")

    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model, shape


def main():
    parser = argparse.ArgumentParser(description="IF5152 Lab 1: PyTorch -> ONNX Export & Verification")
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v3_small",
        choices=["mobilenet_v3_small", "resnet18", "ssdlite320_mobilenet_v3_large"],
    )
    parser.add_argument("--weights", type=str, default=None, help="Optional custom .pth checkpoint")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    os.makedirs(DEFAULT_CHECKPOINT_DIR, exist_ok=True)
    model, input_shape = load_pretrained_model(args.model, args.weights)
    dummy_input = torch.randn(*input_shape)

    onnx_path = os.path.join(DEFAULT_CHECKPOINT_DIR, f"{args.model}.onnx")
    print(f"[1/4] Exporting '{args.model}' to {onnx_path} (opset={args.opset})...")

    is_detector = args.model.startswith("ssdlite")
    output_names = ["boxes", "scores", "labels"] if is_detector else ["logits"]
    dynamic_axes = {"images": {0: "batch_size"}}
    if not is_detector:
        dynamic_axes["logits"] = {0: "batch_size"}

    export_to_onnx(
        model,
        dummy_input,
        onnx_path,
        opset_version=args.opset,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    print("[2/4] Validating ONNX graph structure...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX graph is structurally valid!")

    try:
        from onnxsim import simplify

        print("[3/4] Simplifying ONNX graph with onnxsim...")
        model_simp, ok = simplify(onnx_model)
        if ok:
            onnx.save(model_simp, onnx_path)
            print("  Graph simplified successfully.")
    except ImportError:
        print("[3/4] onnxsim not installed — skipping optional simplification.")

    print("[4/4] Verifying numerical equivalence (PyTorch vs. ONNX Runtime)...")
    with torch.no_grad():
        pt_raw = model(dummy_input)
        if is_detector:
            pt_out = pt_raw[0]["boxes"].numpy()
        else:
            pt_out = pt_raw.numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"images": dummy_input.numpy()})[0]
    np.testing.assert_allclose(pt_out, ort_out, rtol=1e-3, atol=1e-4)
    print(f"  Numerical equivalence verified! Max abs diff: {np.max(np.abs(pt_out - ort_out)):.2e}")
    print(f"  Exported ONNX Size: {get_file_size_mb(onnx_path):.2f} MB -> {onnx_path}")


if __name__ == "__main__":
    main()
