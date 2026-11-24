"""
Session 2 — Lab 2: Production-Style FastAPI Vision Inference Server.
Loads the exported ONNX model once at startup via FastAPI lifespan,
preprocesses uploaded images with Pillow, and serves predictions via POST /predict.

Run locally:
    uvicorn 04_fastapi_server:app --host 0.0.0.0 --port 8000
Interactive Swagger UI:
    http://localhost:8000/docs
"""

import io
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel
from torchvision import models

from utils import DEFAULT_CHECKPOINT_DIR, export_to_onnx

ONNX_PATH = os.environ.get(
    "IF5152_ONNX_PATH",
    os.path.join(DEFAULT_CHECKPOINT_DIR, "mobilenet_v3_small.onnx"),
)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
CATEGORIES = models.MobileNet_V3_Small_Weights.DEFAULT.meta["categories"]

ml_state: dict = {}


def _ensure_default_onnx(onnx_path: str) -> None:
    if os.path.exists(onnx_path):
        return
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).eval()
    dummy = torch.randn(1, 3, 224, 224)
    export_to_onnx(
        model,
        dummy,
        onnx_path,
        opset_version=17,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_default_onnx(ONNX_PATH)
    providers = [
        p for p in ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
        if p in ort.get_available_providers()
    ]
    ml_state["session"] = ort.InferenceSession(ONNX_PATH, providers=providers)
    ml_state["input_name"] = ml_state["session"].get_inputs()[0].name
    ml_state["providers"] = ml_state["session"].get_providers()
    yield
    ml_state.clear()


app = FastAPI(
    title="IF5152 Computer Vision Inference API",
    description="ONNX Runtime + FastAPI deployment server for Week 14 Lab 2.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionItem(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    filename: str
    inference_time_ms: float
    top_predictions: list[PredictionItem]


def preprocess_image(image_bytes: bytes, size: int = 224) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(np.expand_dims(arr, axis=0))  # (1, 3, 224, 224)


def _run_inference(tensor: np.ndarray) -> np.ndarray:
    session = ml_state["session"]
    return session.run(None, {ml_state["input_name"]: tensor})[0][0]


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": os.path.basename(ONNX_PATH),
        "providers": ml_state.get("providers", []),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit.")

    try:
        tensor = preprocess_image(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}") from exc

    t0 = time.perf_counter()
    logits = await run_in_threadpool(_run_inference, tensor)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = _softmax(logits)
    top_idx = np.argsort(probs)[::-1][:3]
    top_preds = [
        PredictionItem(label=CATEGORIES[int(i)], confidence=round(float(probs[i]), 4))
        for i in top_idx
    ]

    return PredictResponse(
        filename=file.filename or "uploaded_image.jpg",
        inference_time_ms=round(latency_ms, 2),
        top_predictions=top_preds,
    )
