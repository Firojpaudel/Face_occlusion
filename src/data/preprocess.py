"""
Runs RetinaFace face detector on all raw source images.
Extracts face crops with a margin (25% on all sides) to capture beanies, caps, helmets, and scarves.
Resizes crops to 224x224 and saves them.
"""
import os
import cv2
import yaml
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_face_crops(
    image_paths: list,
    output_dir: str,
    margin: float = 0.25,
    min_face_px: int = 40,
    face_model: str = "buffalo_s"
) -> list:
    """
    Runs RetinaFace on a list of image paths and extracts face crops.
    Returns: List of dicts with crop metadata: {crop_path, source_path, face_idx, bbox}
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("[ERROR] insightface is required for preprocessing. Install insightface.")
        return []

    # Initialize InsightFace detector
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    app = FaceAnalysis(
        name=face_model,
        allowed_modules=['detection'],
        providers=providers
    )
    # ctx_id=0 if CUDA is available, else -1
    import onnxruntime as ort
    ctx_id = 0 if 'CUDAExecutionProvider' in ort.get_available_providers() else -1
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    records = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(image_paths, desc="Extracting face crops"):
        img_path = Path(img_path)
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        try:
            faces = app.get(img)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

        h, w = img.shape[:2]

        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            bw, bh = x2 - x1, y2 - y1

            # Skip tiny faces
            if bw < min_face_px or bh < min_face_px:
                continue

            # Add margin
            mx = int(bw * margin)
            my = int(bh * margin)
            
            # Clamp to image boundaries
            x1c = max(0, x1 - mx)
            y1c = max(0, y1 - my)
            x2c = min(w, x2 + mx)
            y2c = min(h, y2 + my)

            # Crop and resize
            crop = img[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue

            crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)

            crop_name = f"{img_path.stem}_f{idx}.jpg"
            crop_path = output_dir / crop_name
            cv2.imwrite(str(crop_path), crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 92])

            records.append({
                "crop_path": str(crop_path),
                "source_path": str(img_path),
                "face_idx": idx,
                "bbox_x1": x1c, "bbox_y1": y1c,
                "bbox_x2": x2c, "bbox_y2": y2c,
                "face_w": bw, "face_h": bh,
            })

    return records

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop faces using RetinaFace")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        face_model = cfg.get("face_detector", "buffalo_s")
        margin = cfg.get("face_margin", 0.25)
        min_face_px = cfg.get("min_face_size", 40)
    else:
        face_model = "buffalo_s"
        margin = 0.25
        min_face_px = 40

    print("preprocess.py loaded. Use it as a library or import extract_face_crops.")
