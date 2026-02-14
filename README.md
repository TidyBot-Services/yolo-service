# YOLO Service

Backend service for TidyBot to run YOLO object detection.

## API

`POST /detect`
- Payload: `{"image": "base64_encoded_image"}`
- Response: `{"detections": [...]}`
