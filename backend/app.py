import os
import sys
import time
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Ensure the backend directory is in the import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SSDObjectDetector

# Initialize FastAPI
app = FastAPI(title="InsightSSD API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
try:
    detector = SSDObjectDetector()
except Exception as e:
    print(f"Failed to load SSD Detector: {e}")
    detector = None

@app.get("/api/status")
async def get_status():
    if detector is None:
        return {
            "status": "error",
            "message": "Model not loaded. Ensure download_models.py has run successfully.",
            "backend": "none",
            "classes": []
        }
    
    # Try to determine backend
    backend_str = "CPU"
    try:
        if detector.net.getPreferableBackend() == cv2.dnn.DNN_BACKEND_CUDA:
            backend_str = "CUDA (GPU)"
    except Exception:
        pass
        
    return {
        "status": "active",
        "model_name": "SSD MobileNet V2 (COCO)",
        "backend": backend_str,
        "classes": sorted(list(detector.labels.values()))
    }

@app.post("/api/detect")
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.5, ge=0.0, le=1.0),
    nms_threshold: float = Query(0.4, ge=0.0, le=1.0),
    allowed_classes: str = Query(None),  # Comma-separated list
    model_variant: str = Query("coco")
):
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Object detection model is not loaded on the server."
        )

    # Read bytes and check validity
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    h, w = image.shape[:2]
    
    # Class filters
    classes_filter = None
    if allowed_classes:
        classes_filter = set(cls.strip().lower() for cls in allowed_classes.split(",") if cls.strip())

    # Measure latency
    start_time = time.perf_counter()
    try:
        detections = detector.detect(
            image,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            allowed_classes=classes_filter,
            model_variant=model_variant
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
    latency = (time.perf_counter() - start_time) * 1000  # in milliseconds

    return {
        "width": w,
        "height": h,
        "detections": detections,
        "latency_ms": round(latency, 2)
    }

# Mount static folders for scripts and notebook downloads
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

src_dir = os.path.join(ROOT_DIR, "src")
if os.path.exists(src_dir):
    app.mount("/src", StaticFiles(directory=src_dir), name="src_files")

notebook_dir = os.path.join(ROOT_DIR, "notebook")
if os.path.exists(notebook_dir):
    app.mount("/notebook", StaticFiles(directory=notebook_dir), name="notebook_files")

# Mount compiled production build of frontend
dist_dir = os.path.join(ROOT_DIR, "frontend", "dist")
os.makedirs(dist_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")

