"""
FastAPI production inference server.
Serves liveness checkpoints and maps uploaded images to the two-stage pipeline,
returning detected faces, bounding boxes, and accessory probabilities in JSON.
"""
import os
import cv2
import yaml
import time
import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from src.models.pipeline import AccessoryPipeline, CLASSES

app = FastAPI(
    title="Face Accessory Detection Server",
    description="Low-latency API to detect faces and classify worn accessories.",
    version="1.0.0"
)

# Global pipeline instance initialized on startup
pipeline = None

@app.on_event("startup")
def startup_event():
    global pipeline
    
    # Load config if available to fetch face detector settings
    config_path = "config.yaml"
    face_model = "buffalo_s"
    face_margin = 0.25
    classifier_path = "outputs/accessor_classifier.onnx"
    
    if Path(config_path).exists():
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            face_model = cfg.get("face_detector", "buffalo_s")
            face_margin = cfg.get("face_margin", 0.25)
            
            output_dir = cfg.get("output_dir", "outputs")
            classifier_path = str(Path(output_dir) / "accessor_classifier.onnx")
        except Exception as e:
            print(f"[WARNING] Error reading {config_path}: {e}. Using defaults.")

    print(f"Initializing two-stage pipeline (Detector: {face_model}, Classifier: {classifier_path})...")
    try:
        pipeline = AccessoryPipeline(
            classifier_path=classifier_path,
            face_model=face_model,
            device="auto",
            face_margin=face_margin
        )
        print("✓ Pipeline successfully initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize pipeline: {e}")
        print("FastAPI server running in degraded state. Model inference requests will fail.")

@app.get("/health")
def health():
    status = "ok" if pipeline is not None and pipeline.classifier is not None else "degraded"
    return {"status": status, "timestamp": time.time()}

@app.get("/classes")
def get_classes():
    return {"classes": CLASSES}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is uninitialized or degraded.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    try:
        # Read image contents into memory
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image file: {e}")

    if frame is None:
        raise HTTPException(status_code=400, detail="Decoded image frame is empty or corrupt.")

    # Execute pipeline inference and time it
    t0 = time.perf_counter()
    try:
        detections = pipeline.infer(frame)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution error: {e}")
    latency_ms = (time.perf_counter() - t0) * 1000

    # Format JSON payload response
    return JSONResponse({
        "latency_ms": round(latency_ms, 2),
        "num_faces": len(detections),
        "detections": [
            {
                "bbox": d.bbox,
                "face_confidence": round(d.confidence, 4),
                "accessories": {k: round(v, 4) for k, v in d.accessories.items()},
                "detected_classes": d.detected_classes,
            }
            for d in detections
        ]
    })

# Run with: uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 2
