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
        print(f"{d['class_name']} ({d['confidence']:.2f})")

    # Detect with segmentation masks (polygon format)
    result = client.detect_full("photo.jpg", model="yolov8n-seg", return_masks=True)
    for d in result["detections"]:
        print(f"{d['class_name']}: mask polygons = {len(d.get('mask', []))}")

    # Detect with masks in different formats
    result = client.detect_full("photo.jpg", model="yolov8n-seg", return_masks=True, mask_format="rle")
    result = client.detect_full("photo.jpg", model="yolov8n-seg", return_masks=True, mask_format="bitmap")
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

    def _encode_image(self, image) -> str:
        """Encode image to base64 from file path, bytes, or pass through if already base64."""
        if isinstance(image, (str, Path)):
            return base64.b64encode(Path(image).read_bytes()).decode()
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode()
        return image  # assume already base64

    def detect(
        self,
        image,
        model: str = "yolov8n",
        conf: float = 0.25,
    ) -> list[dict]:
        """
        Run object detection on an image. Returns detections only.

        Args:
            image: File path (str/Path), raw bytes, or base64 string.
            model: Model name. Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, yolov8n-seg, yolov8s-seg
            conf: Confidence threshold (0.0 - 1.0).

        Returns:
            List of detections. Each detection has:
                - bbox: {x1, y1, x2, y2}
                - confidence: float
                - class_id: int
                - class_name: str
        """
        payload = {
            "image": self._encode_image(image),
            "model": model,
            "conf": conf,
            "return_masks": False,
        }
        r = requests.post(f"{self.base_url}/detect", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["detections"]

    def detect_full(
        self,
        image,
        model: str = "yolov8n",
        conf: float = 0.25,
        return_masks: bool = False,
        mask_format: str = "polygon",
    ) -> dict:
        """
        Run detection/segmentation with full metadata.

        Args:
            image: File path (str/Path), raw bytes, or base64 string.
            model: Model name. Use '-seg' models for segmentation (e.g. yolov8n-seg).
            conf: Confidence threshold (0.0 - 1.0).
            return_masks: If True and using a -seg model, include segmentation masks.
            mask_format: 'polygon' (list of [x,y] points), 'rle' (run-length encoding), or 'bitmap' (base64 PNG).

        Returns:
            dict with keys:
                - detections: list of dets (each with bbox, confidence, class_id, class_name, mask)
                - model: str
                - device: str (cuda/cpu)
                - inference_ms: float
                - image_width: int
                - image_height: int
                - has_masks: bool
        """
        payload = {
            "image": self._encode_image(image),
            "model": model,
            "conf": conf,
            "return_masks": return_masks,
            "mask_format": mask_format,
        }
        r = requests.post(f"{self.base_url}/detect", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    client = YOLOClient()
    print("Health:", client.health())
