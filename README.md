# TidyBot YOLO Service

Backend object detection + segmentation service for TidyBot frontend agents. Runs YOLOv8 on GPU (RTX 5090) and exposes results via HTTP API.

## Service URL

```
http://158.130.109.188:8000
```

## Quick Start (Client)

**Only dependency:** `pip install requests`

```python
from client import YOLOClient

client = YOLOClient("http://158.130.109.188:8000")

# Simple detection — returns list of detections
detections = client.detect("photo.jpg")
for d in detections:
    print(f"{d['class_name']} ({d['confidence']:.2f}) at [{d['bbox']['x1']:.0f},{d['bbox']['y1']:.0f},{d['bbox']['x2']:.0f},{d['bbox']['y2']:.0f}]")

# Detection with segmentation masks
result = client.detect_full("photo.jpg", model="yolov8n-seg", return_masks=True)
for d in result["detections"]:
    print(f"{d['class_name']}: {len(d['mask'])} polygon(s)")

# Check service health
print(client.health())
```

## API Reference

### `GET /health`

Returns service status, GPU info, and loaded models.

**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "gpu_memory_mb": 32084,
  "loaded_models": ["yolov8n"],
  "available_models": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x", "yolov8n-seg", "yolov8s-seg"]
}
```

### `POST /detect`

Run object detection or segmentation on a base64-encoded image.

**Request:**
```json
{
  "image": "<base64-encoded-image>",
  "model": "yolov8n",
  "conf": 0.25,
  "return_masks": false,
  "mask_format": "polygon"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | string | required | Base64-encoded JPEG or PNG |
| `model` | string | `"yolov8n"` | Model name (see table below) |
| `conf` | float | `0.25` | Confidence threshold (0-1) |
| `return_masks` | bool | `false` | Include segmentation masks (only with `-seg` models) |
| `mask_format` | string | `"polygon"` | `"polygon"`, `"rle"`, or `"bitmap"` |

**Response:**
```json
{
  "detections": [
    {
      "bbox": {"x1": 10.5, "y1": 20.3, "x2": 200.1, "y2": 350.7},
      "confidence": 0.92,
      "class_id": 0,
      "class_name": "person",
      "mask": [[[x1,y1], [x2,y2], ...]]
    }
  ],
  "model": "yolov8n-seg",
  "device": "cuda",
  "inference_ms": 258.29,
  "image_width": 810,
  "image_height": 1080,
  "has_masks": true
}
```

### Detection Fields

| Field | Type | Description |
|-------|------|-------------|
| `bbox` | object | Bounding box: `{x1, y1, x2, y2}` in pixels |
| `confidence` | float | Detection confidence (0-1) |
| `class_id` | int | COCO class index |
| `class_name` | string | Human-readable label (e.g. "person", "bus") |
| `mask` | varies | Segmentation mask (null if not requested) |

### Mask Formats

| Format | Type | Description |
|--------|------|-------------|
| `polygon` | `list[list[list[float]]]` | List of polygons, each polygon is list of `[x, y]` points |
| `rle` | `{"counts": list[int], "size": [h, w]}` | Run-length encoding |
| `bitmap` | `string` | Base64-encoded PNG of binary mask |

### Available Models

| Model | Type | Speed | Accuracy |
|-------|------|-------|----------|
| `yolov8n` | Detection | Fastest | Good |
| `yolov8s` | Detection | Fast | Better |
| `yolov8m` | Detection | Moderate | Great |
| `yolov8l` | Detection | Slower | Excellent |
| `yolov8x` | Detection | Slowest | Best |
| `yolov8n-seg` | Segmentation | Fast | Good + masks |
| `yolov8s-seg` | Segmentation | Moderate | Better + masks |

### Interactive Docs

Visit `http://158.130.109.188:8000/docs` for auto-generated Swagger UI.

## Discovery

This service is registered in `catalog.json` at:
https://github.com/TidyBot-Services/backend_wishlist/blob/main/catalog.json
