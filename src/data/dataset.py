import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd
import numpy as np

CLASSES = [
    "glasses_clear", "glasses_tinted", "helmet_bike", "helmet_hard",
    "cap_hat", "beanie", "scarf_muffler", "face_mask", "face_shield", "balaclava"
]

def get_train_transforms(image_size=224, aug_cfg=None):
    if aug_cfg is None:
        aug_cfg = {}
        
    p_hflip = aug_cfg.get("horizontal_flip", 0.5)
    p_bc = aug_cfg.get("brightness_contrast", 0.4)
    p_hsv = aug_cfg.get("hue_saturation", 0.3)
    p_blur = aug_cfg.get("motion_blur", 0.2)
    p_persp = aug_cfg.get("perspective", 0.3)
    p_shadow = aug_cfg.get("shadow", 0.2)
    p_dropout = aug_cfg.get("coarse_dropout", 0.2)
    p_jpeg = aug_cfg.get("jpeg_compression", 0.3)
    p_noise = aug_cfg.get("gaussian_noise", 0.2)

    transforms = [
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=p_hflip),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=p_bc),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=p_hsv),
        A.MotionBlur(blur_limit=5, p=p_blur),
        A.Perspective(scale=(0.05, 0.12), p=p_persp),
        # Random shadow is available in newer Albumentations versions, fallback to safe options
        A.RandomShadow(p=p_shadow) if hasattr(A, "RandomShadow") else A.ColorJitter(brightness=0.1, contrast=0.1, p=p_shadow),
        A.CoarseDropout(
            max_holes=4, max_height=30, max_width=30,
            min_holes=1, fill_value=0, p=p_dropout
        ),
        # ImageCompression name compatibility
        A.ImageCompression(quality_range=(50, 95), p=p_jpeg) if hasattr(A, "ImageCompression") else A.JpegCompression(quality_lower=50, p=p_jpeg),
        A.GaussNoise(std_range=(0.01, 0.05), p=p_noise) if hasattr(A, "GaussNoise") else A.GaussianNoise(p=p_noise),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return A.Compose(transforms)

def get_val_transforms(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class AccessoryDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["crop_path"])
        if img is None:
            # Return a blank dummy image if file read fails to prevent crash during training
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        labels = torch.tensor(
            [row[c] for c in CLASSES], dtype=torch.float32
        )
        return img, labels
