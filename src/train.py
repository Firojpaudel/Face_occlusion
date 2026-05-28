"""
Full training script.
- Dynamically calculates BCE pos_weights based on split ratios.
- Mixed-precision (FP16) enabled for T4 GPU speedups.
- Warmup + Cosine learning rate scheduler.
- Checkpoint saving (best and last) for seamless Colab crash-resumption.
"""
import os
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from src.data.dataset import AccessoryDataset, get_train_transforms, get_val_transforms, CLASSES
from src.models.classifier import AccessoryClassifier

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_map(targets, probs):
    """Computes Mean Average Precision (mAP) and per-class APs."""
    aps = []
    for i in range(targets.shape[1]):
        # Skip class if it has no positive labels in the batch/split
        if targets[:, i].sum() == 0:
            aps.append(0.0)
            continue
        ap = average_precision_score(targets[:, i], probs[:, i])
        aps.append(ap)
    return np.mean(aps), aps

def get_pos_weights(train_csv_path, device, multiplier=2.0):
    """
    Computes BCE positive weight multipliers based on class counts.
    Ensures rare classes are weighted heavier in the loss landscape.
    """
    df = pd.read_csv(train_csv_path)
    weights = []
    total_samples = len(df)
    for cls in CLASSES:
        n_pos = df[cls].sum()
        n_neg = total_samples - n_pos
        w = n_neg / max(n_pos, 1)
        # Apply scaling multiplier to handle high imbalance classes
        w = min(w * multiplier, 20.0)   # Cap weight at 20.0 to prevent gradient explosions
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32).to(device)

def train(config_path="config.yaml"):
    seed_everything(42)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up directories
    data_dir = cfg.get("data_dir", "data")
    output_dir = Path(cfg.get("output_dir", "outputs"))
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_csv = Path(data_dir) / "splits" / "train.csv"
    val_csv = Path(data_dir) / "splits" / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Training split files not found at {train_csv}. Run preprocessing/annotation first.")

    # Datasets and Loaders
    train_ds = AccessoryDataset(str(train_csv), get_train_transforms(cfg["image_size"], cfg.get("aug", {})))
    val_ds = AccessoryDataset(str(val_csv), get_val_transforms(cfg["image_size"]))

    # Colab limits workers to 2 safely. Use 4-8 on high-performance servers.
    num_workers = 2 if "colab" in str(Path.cwd()).lower() or os.environ.get("COLAB_GPU") else 4

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Initialize model
    model = AccessoryClassifier(num_classes=cfg["num_classes"]).to(device)

    # Dynamic pos weights
    pos_weights = get_pos_weights(str(train_csv), device, cfg.get("pos_weight_multiplier", 2.0))
    print(f"Calculated class balance loss weights: {pos_weights.tolist()}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    # Cosine learning rate with warmup
    total_steps = cfg["epochs"] * len(train_loader)
    warmup_steps = cfg["warmup_epochs"] * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # FP16 AMP scaler
    scaler = GradScaler()

    # Load checkpoint if exists to allow resuming
    best_map = 0.0
    start_epoch = 0
    
    best_ckpt_path = ckpt_dir / "best.pth"
    last_ckpt_path = ckpt_dir / "last.pth"

    if last_ckpt_path.exists():
        print(f"Found existing last checkpoint at {last_ckpt_path}. Resuming...")
        try:
            checkpoint = torch.load(last_ckpt_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                best_map = checkpoint.get("best_map", 0.0)
            else:
                # Fallback if checkpoint is only model weights
                model.load_state_dict(checkpoint)
            print(f"Resumed from epoch {start_epoch} (Best validation mAP: {best_map:.4f})")
        except Exception as e:
            print(f"⚠️ Failed to load last checkpoint: {e}. Starting from scratch.")

    # Early stopping config
    patience = 8
    patience_counter = 0

    print(f"Starting training loop from epoch {start_epoch+1} to {cfg['epochs']}...")
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Validation phase
        model.eval()
        all_targets, all_probs = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                with autocast():
                    logits = model(imgs)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_targets.append(labels.numpy())

        targets_val = np.vstack(all_targets)
        probs_val = np.vstack(all_probs)
        mean_ap, per_class_ap = compute_map(targets_val, probs_val)

        epoch_loss = train_loss / len(train_loader)
        print(f"\n[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f} | Val mAP: {mean_ap:.4f}")
        for cls, ap in zip(CLASSES, per_class_ap):
            flag = "⚠️" if ap < 0.60 else "✓"
            print(f"  - {cls:<18} AP: {ap:.3f} {flag}")

        # Save checkpoint dictionary
        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_map": max(mean_ap, best_map)
        }
        torch.save(ckpt_state, last_ckpt_path)

        # Check improvement
        if mean_ap > best_map:
            best_map = mean_ap
            torch.save(model.state_dict(), best_ckpt_path)
            patience_counter = 0
            print(f"  🎉 New best validation mAP: {best_map:.4f}! Saved to {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered after {patience} epochs of no mAP improvement.")
                break

    print(f"\nTraining complete. Best validation mAP: {best_map:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face accessory classifier")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    train(args.config)
