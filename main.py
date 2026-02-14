import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI(title="YOLO Detection Service")

# Load YOLOv8 model (downloads automatically)
model = YOLO("yolov8n.pt")

class DetectionRequest(BaseModel):
    image: str  # Base64 encoded image

@app.post("/detect")
async def detect(request: DetectionRequest):
    try:
        # Decode base64 image
        img_data = base64.b64decode(request.image)
        img = Image.open(io.BytesIO(img_data))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Run inference
        results = model(img_cv)

        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "bbox": box.xyxy[0].tolist(),
                    "conf": float(box.conf[0]),
                    "class": int(box.cls[0]),
                    "name": model.names[int(box.cls[0])]
                })

        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
