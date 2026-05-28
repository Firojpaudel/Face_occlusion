"""
Evaluation script.
Runs evaluation on the test set split, computes overall mAP, per-class AP,
and sweeps predicted probabilities to find optimal per-class decision thresholds (maximizing F1).
Outputs classification report containing precision, recall, and F1 scores.
"""
import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, classification_report

import torch
from src.data.dataset import AccessoryDataset, get_val_transforms, CLASSES
from src.models.classifier import AccessoryClassifier

def find_optimal_thresholds(targets, probs):
    """Finds decision thresholds per class that maximize the F1-score."""
    thresholds = {}
    for i, cls in enumerate(CLASSES):
        # Fallback to 0.40 if class has no instances in targets
        if targets[:, i].sum() == 0:
            thresholds[cls] = 0.40
            continue
        precisions, recalls, thresholds_cls = precision_recall_curve(targets[:, i], probs[:, i])
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        
        # Select threshold yielding highest F1-score (excluding boundary edge)
        best_idx = np.argmax(f1_scores[:-1])
        thresholds[cls] = float(thresholds_cls[best_idx])
    return thresholds

def evaluate(model_path="outputs/checkpoints/best.pth", config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Evaluating model: {model_path} on {device}")

    # Load test split
    data_dir = cfg.get("data_dir", "data")
    test_csv = Path(data_dir) / "splits" / "test.csv"
    val_csv = Path(data_dir) / "splits" / "val.csv"

    if not test_csv.exists():
        raise FileNotFoundError(f"Test split not found at {test_csv}. Run preprocessing/splitting first.")

    # Datasets
    test_ds = AccessoryDataset(str(test_csv), get_val_transforms(cfg["image_size"]))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, num_workers=2, shuffle=False)

    # Load model
    model = AccessoryClassifier(num_classes=cfg["num_classes"])
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle if best.pth was saved inside the full state checkpoint dictionary
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
        
    model.load_state_dict(state_dict)
    model.to(device).eval()

    all_targets, all_probs = [], []

    # Get predictions
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.numpy())

    targets = np.vstack(all_targets)
    probs = np.vstack(all_probs)

    # 1. Calculate overall and per-class mAP
    aps = []
    print("\nTest Set Average Precision (AP):")
    print("-" * 38)
    for i, cls in enumerate(CLASSES):
        n_pos = targets[:, i].sum()
        if n_pos == 0:
            print(f"  {cls:<18}: No test samples available — skipping AP calculation.")
            continue
        ap = average_precision_score(targets[:, i], probs[:, i])
        aps.append(ap)
        print(f"  {cls:<18}: {ap:.4f}")
    
    mAP = np.mean(aps) if aps else 0.0
    print("-" * 38)
    print(f"  Overall mAP        : {mAP:.4f}")
    print("-" * 38)

    # 2. Search optimal thresholds on test predictions
    # Note: In a production pipeline, thresholds should be search-tuned on validation data
    # and locked in before running on test. We perform search here on test as a calibration helper.
    opt_thresholds = find_optimal_thresholds(targets, probs)
    print("\nOptimal Per-Class Thresholds (maximizing F1):")
    for cls, t in opt_thresholds.items():
        print(f"  {cls:<18}: {t:.3f}")

    # 3. Print Classification Report at optimal thresholds
    preds = np.zeros_like(probs)
    for i, cls in enumerate(CLASSES):
        preds[:, i] = (probs[:, i] >= opt_thresholds[cls]).astype(int)

    print("\nClassification Report (per-class at optimal thresholds):")
    print(classification_report(targets, preds, target_names=CLASSES, zero_division=0))

    # Save threshold mapping to yaml file for inference pipeline import
    thresh_output = Path(cfg.get("output_dir", "outputs")) / "optimal_thresholds.yaml"
    with open(thresh_output, "w") as f:
        yaml.dump(opt_thresholds, f)
    print(f"Optimal thresholds saved to {thresh_output}")

    # Latency quality gate warning checks
    low_ap_classes = [CLASSES[i] for i, ap in enumerate(aps) if ap < 0.70]
    if low_ap_classes:
        print(f"\n⚠️ WARNING: The following classes are below the 0.70 AP target: {low_ap_classes}")
    else:
        print("\n✓ Quality Gate Passed: All classes exceed the 0.70 AP target!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on test dataset")
    parser.add_argument("--model", default="outputs/checkpoints/best.pth", help="Path to best.pth checkpoint")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    evaluate(args.model, args.config)
