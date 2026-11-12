"""
app_gradio_dcgan_cifar10.py
===========================
Backwards-compatible entrypoint for Generative Models Studio.
This script now forwards directly to the unified studio:
app_gradio_generative_models_cifar10.py
"""

import argparse
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from app_gradio_generative_models_cifar10 import build_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generative Models Studio (CIFAR-10)")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on ('cpu', 'cuda', 'mps')")
    parser.add_argument("--server-port", type=int, default=7860, help="Port to run the Gradio server on")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    demo = build_app(device_str=args.device)
    demo.launch(server_port=args.server_port, share=args.share)
