# TidyBot YOLO Service

Backend object detection service for TidyBot frontend agents. Runs YOLOv8 models on GPU and exposes results via HTTP API.

## Quick Start

### Server (Backend)
```bash
pip install -r requirements.txt
python main.py
# Service starts on http://0.0.0.0:8000
```

### Client (Frontend Agent)
```python
from client import YOLOClient

client = YOLOClient("http://<backend-host>:8000")

# Check service health
print(client.health())

# Detect objects in an image
detections = client.detect("photo.jpg")
for d in detections:
    print(f"{d['class_name']} ({d['confidence']:.2f})")

# Use a larger model for better accuracy
detections = client.detect("photo.jpg", model="yolov8m", conf=0.5)
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
  "gpu_memory_mb": 32607,
  "loaded_models": ["yolov8n"],
  "available_models": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x", "yolov8n-seg", "yolov8s-seg"]
}
```

### `POST /detect`
Run object detection on a base64-encoded image.

**Request:**
```json
{
  "image": "<base64-encoded-image>",
  "model": "yolov8n",
  "conf": 0.25
}
```

**Response:**
```json
{
  "detections": [
    {
      "bbox": {"x1": 10.5, "y1": 20.3, "x2": 200.1, "y2": 350.7},
      "confidence": 0.92,
      "class_id": 0,
      "class_name": "person"
    }
  ],
  "model": "yolov8n",
  "device": "cuda",
  "inference_ms": 12.34,
  "image_width": 640,
  "image_height": 480
}
```

### Available Models
| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolov8n` | Nano | Fastest | Good |
| `yolov8s` | Small | Fast | Better |
| `yolov8m` | Medium | Moderate | Great |
| `yolov8l` | Large | Slower | Excellent |
| `yolov8x` | XLarge | Slowest | Best |
| `yolov8n-seg` | Nano Seg | Fast | Segmentation |
| `yolov8s-seg` | Small Seg | Moderate | Segmentation |

### Interactive API Docs
Once the service is running, visit `http://<backend-host>:8000/docs` for the auto-generated Swagger UI.
