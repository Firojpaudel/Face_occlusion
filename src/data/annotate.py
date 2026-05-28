"""
Annotation compilation and dataset splitting script.
1. Collects CelebA records (eyeglasses and cap/hat attributes).
2. Runs face detection on LFW and marks crops as clean negatives (all zeros).
3. Parses Roboflow YOLOv8 labels, runs face detection on images, maps bounding box classes.
4. (Optional) Auto-labels unlabeled crops using CLIP zero-shot classification.
5. Splits dataset into 80% train, 10% val, 10% test.
"""
import os
import cv2
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

CLASSES = [
    "glasses_clear", "glasses_tinted", "helmet_bike", "helmet_hard",
    "cap_hat", "beanie", "scarf_muffler", "face_mask", "face_shield", "balaclava"
]

RF_CLASS_MAP = {
    "mask": "face_mask",
    "face_mask": "face_mask",
    "helmet": "helmet_hard",
    "hard_hat": "helmet_hard",
    "hardhat": "helmet_hard",
    "bicycle_helmet": "helmet_bike",
    "bike_helmet": "helmet_bike",
    "glasses": "glasses_clear",
    "sunglasses": "glasses_tinted",
    "cap": "cap_hat",
    "hat": "cap_hat",
}

def empty_labels():
    return {c: 0 for c in CLASSES}

def load_face_detector(face_model):
    from insightface.app import FaceAnalysis
    import onnxruntime as ort
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    app = FaceAnalysis(name=face_model, allowed_modules=['detection'], providers=providers)
    ctx_id = 0 if 'CUDAExecutionProvider' in ort.get_available_providers() else -1
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app

def process_celeba(data_dir, all_rows):
    celeba_labels_csv = Path(data_dir) / "raw" / "celeba" / "celeba_labels.csv"
    if not celeba_labels_csv.exists():
        print("[SKIP] CelebA labels CSV not found.")
        return

    print("Processing CelebA annotations...")
    df = pd.read_csv(celeba_labels_csv)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="CelebA"):
        labels = empty_labels()
        labels["glasses_clear"] = int(row["glasses_clear"])
        labels["cap_hat"] = int(row["cap_hat"])
        
        r = {"crop_path": row["image_path"]}
        r.update(labels)
        all_rows.append(r)

def process_lfw(data_dir, face_app, all_rows, max_negatives=8000):
    lfw_raw_dir = Path(data_dir) / "raw" / "lfw"
    if not lfw_raw_dir.exists():
        print("[SKIP] LFW directory not found.")
        return

    print("Extracting clean negatives from LFW...")
    crops_dir = Path(data_dir) / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    lfw_images = list(lfw_raw_dir.rglob("*.jpg"))[:max_negatives]
    
    from src.data.preprocess import extract_face_crops
    records = extract_face_crops(
        lfw_images, 
        str(crops_dir), 
        margin=0.25, 
        min_face_px=40,
        face_model=face_app.name
    )

    for rec in records:
        r = {"crop_path": rec["crop_path"]}
        r.update(empty_labels())
        all_rows.append(r)

def process_roboflow(data_dir, face_app, all_rows):
    rf_dir = Path(data_dir) / "raw" / "roboflow"
    if not rf_dir.exists():
        print("[SKIP] Roboflow raw directory not found.")
        return

    crops_dir = Path(data_dir) / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    print("Processing Roboflow YOLOv8 folders...")
    for project_folder in rf_dir.iterdir():
        if not project_folder.is_dir():
            continue

        print(f"  Project: {project_folder.name}")
        yaml_path = project_folder / "data.yaml"
        if not yaml_path.exists():
            continue

        with open(yaml_path) as f:
            data_yaml = yaml.safe_load(f)
        rf_classes = data_yaml.get("names", [])

        # Process train/val/test splits in Roboflow export
        for split in ["train", "valid", "test"]:
            img_dir = project_folder / split / "images"
            lbl_dir = project_folder / split / "labels"
            if not img_dir.exists():
                continue

            img_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            for img_path in tqdm(img_paths, desc=f"    {split} split"):
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                labels = empty_labels()

                if lbl_path.exists():
                    with open(lbl_path) as f:
                        for line in f:
                            cls_idx = int(line.split()[0])
                            if cls_idx < len(rf_classes):
                                rf_cls = rf_classes[cls_idx].lower().replace(" ", "_")
                                our_cls = RF_CLASS_MAP.get(rf_cls)
                                if our_cls:
                                    labels[our_cls] = 1

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                try:
                    faces = face_app.get(img)
                except Exception:
                    continue

                h, w = img.shape[:2]
                for idx, face in enumerate(faces):
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    bw, bh = x2 - x1, y2 - y1
                    if bw < 40 or bh < 40:
                        continue

                    # Pad and crop
                    mx, my = int(bw * 0.25), int(bh * 0.25)
                    x1c = max(0, x1 - mx)
                    y1c = max(0, y1 - my)
                    x2c = min(w, x2 + mx)
                    y2c = min(h, y2 + my)

                    crop = img[y1c:y2c, x1c:x2c]
                    if crop.size == 0:
                        continue

                    crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
                    name = f"rf_{img_path.stem}_f{idx}.jpg"
                    out_path = crops_dir / name
                    cv2.imwrite(str(out_path), crop_resized)

                    r = {"crop_path": str(out_path)}
                    r.update(labels)
                    all_rows.append(r)

def run_clip_autolabel(ann_path, data_dir, threshold=0.30):
    try:
        import torch
        import clip
        from PIL import Image
    except ImportError:
        print("[SKIP] CLIP/torch is not installed. Install them to run auto-labeling.")
        return

    df = pd.read_csv(ann_path)
    # Target all-zero (unlabeled) crops
    unlabeled_mask = df[CLASSES].sum(axis=1) == 0
    unlabeled_paths = df[unlabeled_mask]["crop_path"].tolist()
    
    if len(unlabeled_paths) == 0:
        print("No unlabeled crops found. Skipping CLIP auto-labeling.")
        return

    print(f"Running CLIP zero-shot classification on {len(unlabeled_paths)} unlabeled crops...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    CLIP_PROMPTS = {
        "glasses_clear":  ["person wearing clear prescription eyeglasses", "face with glasses"],
        "glasses_tinted": ["person wearing dark sunglasses", "face with tinted sunglasses"],
        "helmet_bike":    ["cyclist wearing bicycle helmet", "person with bike helmet"],
        "helmet_hard":    ["construction worker wearing hard hat", "person wearing safety helmet"],
        "cap_hat":        ["person wearing baseball cap", "face with a hat or cap"],
        "beanie":         ["person wearing beanie or knit hat", "face with wool cap"],
        "scarf_muffler":  ["person with scarf around face or neck", "face with muffler"],
        "face_mask":      ["person wearing medical face mask", "face with surgical mask"],
        "face_shield":    ["person wearing transparent face shield", "clear face visor"],
        "balaclava":      ["person wearing balaclava ski mask", "face covered by balaclava"],
    }
    NEG_PROMPTS = ["bare face with no accessories", "plain face"]

    results = {}
    batch_size = 128
    
    for i in tqdm(range(0, len(unlabeled_paths), batch_size), desc="CLIP batch"):
        batch_paths = unlabeled_paths[i:i+batch_size]
        imgs = []
        valid_paths = []
        for p in batch_paths:
            try:
                img = clip_preprocess(Image.open(p)).unsqueeze(0)
                imgs.append(img)
                valid_paths.append(p)
            except Exception:
                continue
        if not imgs:
            continue
        
        imgs_t = torch.cat(imgs).to(device)
        with torch.no_grad():
            img_feats = clip_model.encode_image(imgs_t)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)

            for cls, pos_prompts in CLIP_PROMPTS.items():
                all_prompts = pos_prompts + NEG_PROMPTS
                text = clip.tokenize(all_prompts).to(device)
                txt_feats = clip_model.encode_text(text)
                txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
                
                sims = (img_feats @ txt_feats.T).cpu().numpy()
                for j, path in enumerate(valid_paths):
                    if path not in results:
                        results[path] = empty_labels()
                    pos_score = sims[j, :len(pos_prompts)].max()
                    results[path][cls] = 1 if pos_score > threshold else 0

    # Write back
    for path, labels in results.items():
        mask = df["crop_path"] == path
        for cls, val in labels.items():
            df.loc[mask, cls] = val

    df.to_csv(ann_path, index=False)
    print("✓ CLIP Auto-labeling complete.")

def split_and_save(ann_path, splits_dir):
    Path(splits_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ann_path)

    # Filter out missing crop files
    existing_mask = df["crop_path"].apply(lambda p: os.path.exists(p))
    df = df[existing_mask].reset_index(drop=True)
    print(f"Total valid crop files found: {len(df)}")

    # Class distribution check
    print("\nClass distribution:")
    for cls in CLASSES:
        n = df[cls].sum()
        flag = " ⚠️ LOW" if n < 500 else " ✓"
        print(f"  {cls:<18} {n:>6}{flag}")

    # Split into 80/10/10
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df.to_csv(Path(splits_dir) / "train.csv", index=False)
    val_df.to_csv(Path(splits_dir) / "val.csv", index=False)
    test_df.to_csv(Path(splits_dir) / "test.csv", index=False)
    print(f"\nSplits saved in {splits_dir}:")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

def process_openimages(data_dir, face_app, all_rows):
    oi_raw_dir = Path(data_dir) / "raw" / "open-images-v7"
    if not oi_raw_dir.exists():
        print("[SKIP] OpenImages raw directory not found.")
        return

    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        print("[SKIP] fiftyone is not installed. Install it to process OpenImages.")
        return

    print("Processing OpenImages annotations...")
    
    # Configure global zoo directory so fiftyone finds the dataset
    fo.config.dataset_zoo_dir = str(Path(data_dir) / "raw")

    try:
        oi_classes = [
            "Human face", "Sunglasses", "Helmet", "Hat",
            "Scarf", "Bicycle helmet", "Goggles"
        ]
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=oi_classes,
        )
    except Exception as e:
        print(f"  ⚠️ Could not load FiftyOne dataset: {e} — skipping.")
        return

    crops_dir = Path(data_dir) / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    OI_MAP = {
        "Sunglasses": "glasses_tinted",
        "Helmet": "helmet_hard",
        "Bicycle helmet": "helmet_bike",
        "Hat": "cap_hat",
        "Scarf": "scarf_muffler",
        "Goggles": "glasses_clear",
    }

    print(f"  Found {len(dataset)} samples in OpenImages v7.")
    for sample in tqdm(dataset, desc="OpenImages"):
        img_path = Path(sample.filepath)
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        detections = []
        if sample.detections and sample.detections.detections:
            detections = sample.detections.detections

        try:
            faces = face_app.get(img)
        except Exception:
            continue

        h, w = img.shape[:2]

        for idx, face in enumerate(faces):
            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
            fbw, fbh = fx2 - fx1, fy2 - fy1
            if fbw < 40 or fbh < 40:
                continue

            labels = empty_labels()

            for det in detections:
                label = det.label
                our_cls = OI_MAP.get(label)
                if not our_cls:
                    continue

                rx, ry, rw, rh = det.bounding_box
                dx1, dy1 = int(rx * w), int(ry * h)
                dx2, dy2 = int((rx + rw) * w), int((ry + rh) * h)

                ix1 = max(fx1, dx1)
                iy1 = max(fy1, dy1)
                ix2 = min(fx2, dx2)
                iy2 = min(fy2, dy2)

                inter_w = max(0, ix2 - ix1)
                inter_h = max(0, iy2 - iy1)
                inter_area = inter_w * inter_h
                det_area = (dx2 - dx1) * (dy2 - dy1)

                if det_area > 0 and (inter_area / det_area) > 0.4:
                    labels[our_cls] = 1

            mx, my = int(fbw * 0.25), int(fbh * 0.25)
            x1c = max(0, fx1 - mx)
            y1c = max(0, fy1 - my)
            x2c = min(w, fx2 + mx)
            y2c = min(h, fy2 + my)

            crop = img[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue

            crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
            name = f"oi_{img_path.stem}_f{idx}.jpg"
            out_path = crops_dir / name
            cv2.imwrite(str(out_path), crop_resized)

            r = {"crop_path": str(out_path)}
            r.update(labels)
            all_rows.append(r)

def main():
    parser = argparse.ArgumentParser(description="Compile annotations and build dataset splits")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--autolabel", action="store_true", help="Run CLIP auto-labeling on unlabeled samples")
    args = parser.parse_args()

    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        data_dir = cfg.get("data_dir", "data")
        face_model = cfg.get("face_detector", "buffalo_s")
    else:
        data_dir = "data"
        face_model = "buffalo_s"

    ann_path = Path(data_dir) / "annotations.csv"
    splits_dir = Path(data_dir) / "splits"

    all_rows = []
    face_app = load_face_detector(face_model)

    process_celeba(data_dir, all_rows)
    process_lfw(data_dir, face_app, all_rows)
    process_roboflow(data_dir, face_app, all_rows)
    process_openimages(data_dir, face_app, all_rows)

    if not all_rows:
        print("[ERROR] No annotations were gathered. Check raw data paths.")
        return

    # Save compiled CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(ann_path, index=False)
    print(f"\nSaved compiled annotations to {ann_path} ({len(df)} rows)")

    if args.autolabel:
        run_clip_autolabel(str(ann_path), data_dir)

    split_and_save(str(ann_path), str(splits_dir))

if __name__ == "__main__":
    main()
