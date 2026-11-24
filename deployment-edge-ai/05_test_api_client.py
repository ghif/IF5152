"""
Session 2 — Lab 2 Verification Client: Tests GET /health and POST /predict.
Runs in-process via FastAPI TestClient by default, or against a live server with --url.
"""

import argparse
from importlib import import_module
import os
from fastapi.testclient import TestClient
import httpx

from utils import ensure_sample_images


def test_in_process(sample_paths: list[str]) -> None:
    server_mod = import_module("04_fastapi_server")
    with TestClient(server_mod.app) as client:
        r_health = client.get("/health")
        assert r_health.status_code == 200, r_health.text
        print("[/health]", r_health.json())

        for img_path in sample_paths:
            with open(img_path, "rb") as f:
                r_pred = client.post(
                    "/predict",
                    files={"file": (os.path.basename(img_path), f, "image/jpeg")},
                )
            assert r_pred.status_code == 200, r_pred.text
            print("[/predict]", r_pred.json())


def test_live_server(base_url: str, sample_paths: list[str]) -> None:
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        r_health = client.get("/health")
        r_health.raise_for_status()
        print("[/health]", r_health.json())

        for img_path in sample_paths:
            with open(img_path, "rb") as f:
                r_pred = client.post(
                    "/predict",
                    files={"file": (os.path.basename(img_path), f, "image/jpeg")},
                )
            r_pred.raise_for_status()
            print("[/predict]", r_pred.json())


def main():
    parser = argparse.ArgumentParser(description="IF5152 Lab 2: Test FastAPI Inference Server")
    parser.add_argument("--url", type=str, default=None, help="Live server URL (e.g. http://localhost:8000)")
    args = parser.parse_args()

    samples = ensure_sample_images(count=3)
    if args.url:
        test_live_server(args.url, samples)
    else:
        test_in_process(samples)
    print("\nAll FastAPI endpoint checks passed!")


if __name__ == "__main__":
    main()
