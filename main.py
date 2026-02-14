"""
YOLO Detection Service — TidyBot Backend
Hosted on FastAPI. Exposes object detection via HTTP API.
"""

import base64
import io
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from ultralytics import YOLO

# ─── Model Registry ───────────────────────────────────────────────
MODELS = {}
AVAILABLE_MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
    "yolov8n-seg": "yolov8n-seg.pt",
    "yolov8s-seg": "yolov8s-seg.pt",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "yolov8n"


def get_model(name: str) -> YOLO:
    """Lazy-load and cache models."""
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(AVAILABLE_MODELS.keys())}")
    if name not in MODELS:
        MODELS[name] = YOLO(AVAILABLE_MODELS[name])
        MODELS[name].to(DEVICE)
    return MODELS[name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load default model on startup
    print(f"Loading default model '{DEFAULT_MODEL}' on {DEVICE}...")
    get_model(DEFAULT_MODEL)
    print("Ready.")
    yield
    MODELS.clear()


# ─── FastAPI App ──────────────────────────────────────────────────
app = FastAPI(
    title="TidyBot YOLO Service",
    description="Backend object detection service for TidyBot frontend agents.",
    version="0.1.0",
    lifespan=lifespan,
)


# ─── Schemas ──────────────────────────────────────────────────────
class DetectRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image (JPEG or PNG)")
    model: str = Field(DEFAULT_MODEL, description="Model name (e.g. yolov8n, yolov8s, yolov8n-seg)")
    conf: float = Field(0.25, description="Confidence threshold (0-1)")

class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class Detection(BaseModel):
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str

class DetectResponse(BaseModel):
    detections: list[Detection]
    model: str
    device: str
    inference_ms: float
    image_width: int
    image_height: int

class HealthResponse(BaseModel):
    status: str
    device: str
    gpu_name: Optional[str]
    gpu_memory_mb: Optional[int]
    loaded_models: list[str]
    available_models: list[str]


# ─── Endpoints ────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """Check service health and GPU status."""
    gpu_name = None
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
    return HealthResponse(
        status="ok",
        device=DEVICE,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_mem,
        loaded_models=list(MODELS.keys()),
        available_models=list(AVAILABLE_MODELS.keys()),
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    """Run YOLO object detection on a base64-encoded image."""
    try:
        model = get_model(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        img_data = base64.b64decode(request.image)
        img = Image.open(io.BytesIO(img_data))
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    h, w = img_np.shape[:2]

    t0 = time.perf_counter()
    results = model(img_np, conf=request.conf, verbose=False)
    inference_ms = (time.perf_counter() - t0) * 1000

    detections = []
    for r in results:
        for box in r.boxes:
            coords = box.xyxy[0].tolist()
            detections.append(Detection(
                bbox=BBox(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]),
                confidence=float(box.conf[0]),
                class_id=int(box.cls[0]),
                class_name=model.names[int(box.cls[0])],
            ))

    return DetectResponse(
        detections=detections,
        model=request.model,
        device=DEVICE,
        inference_ms=round(inference_ms, 2),
        image_width=w,
        image_height=h,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
