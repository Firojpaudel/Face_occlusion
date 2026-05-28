"""
Dataset acquisition module.
Downloads:
1. OpenImages v7 (Sunglasses, Helmet, Hat, Scarf, Medical mask, etc.) via FiftyOne.
2. Curated public Roboflow datasets.
3. CelebA (for clean face baseline + eyeglasses/hat labels) via Hugging Face.
4. LFW (clean face baseline, all negative labels).
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import urllib.request
import tarfile
import yaml

def download_openimages(data_dir, max_samples=5000):
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        print("[SKIP] fiftyone is not installed. Install it to download OpenImages.")
        return

    oi_dir = Path(data_dir) / "raw" / "open-images-v7"
    done_flag = oi_dir / "done.flag"
    if done_flag.exists():
        print("✓ OpenImages already downloaded, skipping.")
        return

    print(f"Downloading OpenImages v7 (max {max_samples} samples)...")
    print("  Note: This downloads a large metadata CSV first (~2GB), then images.")
    print("  Expected time: 15-40 min on Colab. If too slow, re-run with --skip-openimages.")
    
    # Configure FiftyOne's zoo directory globally to point to our local data/raw
    # This is a sure fix for 'TypeError: build_dataset_importer() got multiple values for keyword argument 'dataset_dir''
    fo.config.dataset_zoo_dir = str(Path(data_dir) / "raw")

    # Only classes that exist in FiftyOne's OpenImages v7 vocabulary.
    # "Medical mask", "Motorcycle helmet", "Hard hat" do NOT exist in OI v7.
    # Those classes are covered by Roboflow datasets instead.
    oi_classes = [
        "Human face", "Sunglasses", "Helmet", "Hat",
        "Scarf", "Bicycle helmet", "Goggles"
    ]

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=oi_classes,
        max_samples=max_samples,
    )
    oi_dir.mkdir(parents=True, exist_ok=True)
    done_flag.touch()
    print(f"✓ OpenImages download complete. Loaded {len(dataset)} images.")

def download_roboflow(data_dir):
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[SKIP] roboflow is not installed. Install it to download Roboflow datasets.")
        return

    api_key = os.getenv("ROBOFLOW_API_KEY", "YOUR_FREE_API_KEY_HERE")
    if api_key == "YOUR_FREE_API_KEY_HERE":
        print("[WARNING] ROBOFLOW_API_KEY env var not set. Roboflow download might fail.")
        print("Set it or get a free key at roboflow.com")

    rf_dir = Path(data_dir) / "raw" / "roboflow"
    rf_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("roboflow-100", "face-mask-detection-tmnwx", 4),
        ("roboflow-universe-projects", "hard-hat-detection-pv7it", 1),
        ("roboflow-universe-projects", "bicycle-helmet-detection", 2),
        ("roboflow-universe-projects", "glasses-detection-ag0lt", 1),
    ]

    rf = Roboflow(api_key=api_key)
    for workspace, project, version in datasets:
        out_dir = rf_dir / project
        if out_dir.exists():
            print(f"  ✓ {project} already downloaded, skipping.")
            continue
        try:
            print(f"Downloading Roboflow dataset: {project} (v{version})...")
            proj = rf.workspace(workspace).project(project)
            proj.version(version).download("yolov8", location=str(out_dir))
            print(f"  ✓ Downloaded {project}")
        except Exception as e:
            print(f"  ⚠️ Failed to download {project}: {e} — skipping.")

def download_celeba(data_dir, max_samples=60000):
    """Downloads CelebA via HuggingFace for stability."""
    celeba_dir = Path(data_dir) / "raw" / "celeba"
    labels_csv = celeba_dir / "celeba_labels.csv"

    if labels_csv.exists():
        print("✓ CelebA dataset already ready, skipping.")
        return

    try:
        from datasets import load_dataset
        import pandas as pd
    except ImportError:
        print("[SKIP] datasets/pandas is not installed. Install them to download CelebA.")
        return

    celeba_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CelebA from Hugging Face (first {max_samples} samples)...")
    try:
        celeba = load_dataset("tpremoli/CelebA-attrs", split="train", streaming=False)
    except Exception as e:
        print(f"⚠️ CelebA download failed: {e}. Trying secondary method/skipping.")
        return

    rows = []
    for i, sample in enumerate(celeba):
        if i >= max_samples:
            break
        img_path = celeba_dir / f"img_{i:06d}.jpg"
        sample["image"].save(img_path)
        
        # Attribute mappings: Eyeglasses -> glasses_clear, Wearing_Hat -> cap_hat
        row = {
            "image_path": str(img_path),
            "glasses_clear": 1 if sample.get("Eyeglasses", 0) == 1 else 0,
            "cap_hat": 1 if sample.get("Wearing_Hat", 0) == 1 else 0,
        }
        rows.append(row)
        if i > 0 and i % 10000 == 0:
            print(f"  Saved {i}/{max_samples} images...")

    df = pd.DataFrame(rows)
    df.to_csv(labels_csv, index=False)
    print(f"✓ CelebA: saved {len(df)} images and labels to {labels_csv}")

def download_lfw(data_dir):
    lfw_dir = Path(data_dir) / "raw" / "lfw"
    if lfw_dir.exists() and any(lfw_dir.rglob("*.jpg")):
        print("✓ LFW already downloaded, skipping.")
        return

    lfw_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading LFW dataset (clean faces baseline) via sklearn...")
    try:
        from sklearn.datasets import fetch_lfw_people
        # This will download and extract LFW into lfw_dir/lfw_home/lfw_funneled/
        fetch_lfw_people(data_home=str(lfw_dir), color=True, download_if_missing=True)
        print("✓ LFW downloaded and extracted.")
    except Exception as e:
        print(f"⚠️ LFW download failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download raw datasets for Face Accessory Detector")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--skip-openimages", action="store_true",
                        help="Skip OpenImages download (slow). CelebA+Roboflow+LFW already give 88k+ images.")
    parser.add_argument("--skip-roboflow", action="store_true",
                        help="Skip Roboflow download (requires API key).")
    parser.add_argument("--oi-samples", type=int, default=5000,
                        help="Max OpenImages samples to download (default: 5000).")
    args = parser.parse_args()

    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        data_dir = cfg.get("data_dir", "data")
    else:
        data_dir = "data"

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    print("Starting Phase 1 -- Dataset Acquisition...")
    print("Data sources: OpenImages (supplementary), Roboflow (accessory labels),")
    print("              CelebA (glasses/hat + negatives), LFW (clean negatives)")
    print()

    if args.skip_openimages:
        print("[SKIP] OpenImages skipped via --skip-openimages flag.")
    else:
        download_openimages(data_dir, max_samples=args.oi_samples)

    if args.skip_roboflow:
        print("[SKIP] Roboflow skipped via --skip-roboflow flag.")
    else:
        download_roboflow(data_dir)

    download_celeba(data_dir)
    download_lfw(data_dir)
    print("\nDataset acquisition phase completed.")

if __name__ == "__main__":
    main()
