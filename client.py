"""
TidyBot YOLO Service — Python Client SDK

Usage:
    from client import YOLOClient

    client = YOLOClient("http://<backend-host>:8000")

    # Check service health
    health = client.health()
    print(health)

    # Detect objects in an image file
    detections = client.detect("photo.jpg")
    for d in detections:
        print(f"{d['class_name']} ({d['confidence']:.2f}) at [{d['bbox']['x1']:.0f}, {d['bbox']['y1']:.0f}, {d['bbox']['x2']:.0f}, {d['bbox']['y2']:.0f}]")

    # Detect with a specific model and confidence threshold
    detections = client.detect("photo.jpg", model="yolov8s", conf=0.5)
"""

import base64
import requests
from pathlib import Path
from typing import Optional


class YOLOClient:
    """Client SDK for the TidyBot YOLO Detection Service."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        """
        Args:
            base_url: The URL where the YOLO service is hosted (e.g. http://192.168.1.100:8000)
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        """
        Check service health and GPU status.

        Returns:
            dict with keys: status, device, gpu_name, gpu_memory_mb, loaded_models, available_models
        """
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def detect(
        self,
        image: str | bytes | Path,
        model: str = "yolov8n",
        conf: float = 0.25,
    ) -> list[dict]:
        """
        Run object detection on an image.

        Args:
            image: File path (str/Path), raw bytes, or base64 string.
            model: Model name. Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, yolov8n-seg, yolov8s-seg
            conf: Confidence threshold (0.0 - 1.0).

        Returns:
            List of detections, each with: bbox, confidence, class_id, class_name
        """
        # Encode image to base64
        if isinstance(image, (str, Path)):
            image_b64 = base64.b64encode(Path(image).read_bytes()).decode()
        elif isinstance(image, bytes):
            image_b64 = base64.b64encode(image).decode()
        else:
            image_b64 = image  # assume already base64

        payload = {
            "image": image_b64,
            "model": model,
            "conf": conf,
        }

        r = requests.post(f"{self.base_url}/detect", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["detections"]

    def detect_raw(
        self,
        image: str | bytes | Path,
        model: str = "yolov8n",
        conf: float = 0.25,
    ) -> dict:
        """
        Same as detect() but returns the full response including metadata.

        Returns:
            dict with keys: detections, model, device, inference_ms, image_width, image_height
        """
        if isinstance(image, (str, Path)):
            image_b64 = base64.b64encode(Path(image).read_bytes()).decode()
        elif isinstance(image, bytes):
            image_b64 = base64.b64encode(image).decode()
        else:
            image_b64 = image

        payload = {
            "image": image_b64,
            "model": model,
            "conf": conf,
        }

        r = requests.post(f"{self.base_url}/detect", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    # Quick test
    client = YOLOClient()
    print("Health:", client.health())
