"""
YOLO Detection Service — TidyBot Backend
Hosted on FastAPI. Exposes object detection + segmentation via HTTP API.
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
    print(f"Loading default model '{DEFAULT_MODEL}' on {DEVICE}...")
    get_model(DEFAULT_MODEL)
    print("Ready.")
    yield
    MODELS.clear()


# ─── FastAPI App ──────────────────────────────────────────────────
app = FastAPI(
    title="TidyBot YOLO Service",
    description="Backend object detection + segmentation service for TidyBot frontend agents.",
    version="0.2.0",
    lifespan=lifespan,
)


# ─── Schemas ──────────────────────────────────────────────────────
class DetectRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image (JPEG or PNG)")
    model: str = Field(DEFAULT_MODEL, description="Model name (e.g. yolov8n, yolov8n-seg)")
    conf: float = Field(0.25, description="Confidence threshold (0-1)")
    return_masks: bool = Field(False, description="Return segmentation masks (only for -seg models)")
    mask_format: str = Field("polygon", description="Mask format: 'polygon' (list of points), 'rle' (run-length encoded), or 'bitmap' (base64 PNG)")

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
    mask: Optional[list | dict | str] = Field(None, description="Segmentation mask (polygon points, RLE dict, or base64 bitmap)")

class DetectResponse(BaseModel):
    detections: list[Detection]
    model: str
    device: str
    inference_ms: float
    image_width: int
    image_height: int
    has_masks: bool = False

class HealthResponse(BaseModel):
    status: str
    device: str
    gpu_name: Optional[str]
    gpu_memory_mb: Optional[int]
    loaded_models: list[str]
    available_models: list[str]


def mask_to_polygon(mask_np: np.ndarray) -> list[list[float]]:
    """Convert binary mask to polygon contour points."""
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            polygon = contour.squeeze().tolist()
            if isinstance(polygon[0], list):
                polygons.append(polygon)
    return polygons


def mask_to_rle(mask_np: np.ndarray) -> dict:
    """Convert binary mask to run-length encoding."""
    pixels = mask_np.flatten()
    runs = []
    current = 0
    count = 0
    for p in pixels:
        if p == current:
            count += 1
        else:
            runs.append(count)
            current = p
            count = 1
    runs.append(count)
    return {"counts": runs, "size": list(mask_np.shape)}


def mask_to_bitmap_b64(mask_np: np.ndarray) -> str:
    """Convert binary mask to base64-encoded PNG."""
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    _, png_data = cv2.imencode(".png", mask_uint8)
    return base64.b64encode(png_data.tobytes()).decode()


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
    """Run YOLO detection (or segmentation) on a base64-encoded image."""
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
    is_seg_model = "seg" in request.model

    t0 = time.perf_counter()
    results = model(img_np, conf=request.conf, verbose=False)
    inference_ms = (time.perf_counter() - t0) * 1000

    detections = []
    has_masks = False

    for r in results:
        for i, box in enumerate(r.boxes):
            coords = box.xyxy[0].tolist()
            det = Detection(
                bbox=BBox(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]),
                confidence=float(box.conf[0]),
                class_id=int(box.cls[0]),
                class_name=model.names[int(box.cls[0])],
            )

            # Add mask if segmentation model and masks requested
            if request.return_masks and is_seg_model and r.masks is not None:
                has_masks = True
                mask_np = r.masks.data[i].cpu().numpy()
                # Resize mask to original image dimensions
                mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

                if request.mask_format == "polygon":
                    det.mask = mask_to_polygon(mask_resized)
                elif request.mask_format == "rle":
                    det.mask = mask_to_rle(mask_resized.astype(np.uint8))
                elif request.mask_format == "bitmap":
                    det.mask = mask_to_bitmap_b64(mask_resized)
                else:
                    det.mask = mask_to_polygon(mask_resized)

            detections.append(det)

    return DetectResponse(
        detections=detections,
        model=request.model,
        device=DEVICE,
        inference_ms=round(inference_ms, 2),
        image_width=w,
        image_height=h,
        has_masks=has_masks,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
