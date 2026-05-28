import torch
import torch.nn as nn
import torchvision.models as models

class AccessoryClassifier(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        # MobileNetV3-Small: ~2.5MB, ~2ms inference, ideal for low-latency targets
        # We fetch the weights dynamically, using IMAGENET1K_V1
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        # Replace the classification head
        in_features = backbone.classifier[0].in_features   # 576
        backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
            # Note: No Sigmoid here, we use BCEWithLogitsLoss during training.
            # Sigmoid is applied during predict/inference.
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def predict(self, x, threshold=0.4):
        """
        Convenience method for PyTorch model inference.
        Returns:
            probs: probability scores (after Sigmoid)
            preds: binary multi-hot class indicators
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        if isinstance(threshold, (int, float)):
            preds = (probs > threshold).float()
        else:
            # If threshold is a tensor or list of thresholds per class
            thresh_tensor = torch.tensor(threshold, device=x.device, dtype=torch.float32)
            preds = (probs > thresh_tensor).float()
        return probs, preds
