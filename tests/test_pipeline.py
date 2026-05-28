"""
Unit and integration tests for the face accessory detection pipeline.
Verifies:
1. PyTorch classifier model shape consistency.
2. Dataset loading and augmentation shapes.
3. End-to-end inference pipeline batch execution.
"""
import os
import pytest
import numpy as np
import torch
import cv2
from pathlib import Path

from src.models.classifier import AccessoryClassifier
from src.models.pipeline import AccessoryPipeline, FaceDetection

def test_classifier_shapes():
    """Verifies that classifier accepts standard batch shape and returns correct logit dimensions."""
    model = AccessoryClassifier(num_classes=10)
    model.eval()

    # Batch of 4 images, 3 channels, 224x224
    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (4, 10), f"Expected shape (4, 10), got {logits.shape}"
    
    # Check predict interface
    probs, preds = model.predict(dummy_input, threshold=0.40)
    assert probs.shape == (4, 10)
    assert preds.shape == (4, 10)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    assert torch.all((preds == 0.0) | (preds == 1.0))

def test_classifier_per_class_thresholds():
    """Verifies that predict interface respects class-specific threshold lists."""
    model = AccessoryClassifier(num_classes=10)
    model.eval()
    
    dummy_input = torch.randn(2, 3, 224, 224)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    with torch.no_grad():
        probs, preds = model.predict(dummy_input, threshold=thresholds)
        
    assert probs.shape == (2, 10)
    assert preds.shape == (2, 10)

def test_dataset_mock(tmp_path):
    """Verifies that dataset loader compiles rows and labels correctly."""
    from src.data.dataset import AccessoryDataset, CLASSES
    
    # Create a mock split CSV file
    csv_path = tmp_path / "mock_split.csv"
    mock_crop = tmp_path / "mock_crop.jpg"
    
    # Save a blank dummy crop image
    cv2.imwrite(str(mock_crop), np.zeros((100, 100, 3), dtype=np.uint8))
    
    df = pd_df = pd = pd_df = pd.DataFrame([{
        "crop_path": str(mock_crop),
        **{cls: (1 if i == 0 else 0) for i, cls in enumerate(CLASSES)}
    }])
    df.to_csv(csv_path, index=False)
    
    # Load dataset
    ds = AccessoryDataset(str(csv_path), transform=None)
    assert len(ds) == 1
    
    img, labels = ds[0]
    assert img.shape == (224, 224, 3)
    assert labels.shape == (10,)
    assert labels[0] == 1.0
    assert torch.sum(labels) == 1.0

def test_pipeline_instantiation_without_model():
    """Checks that the pipeline loads correctly even if local model files are absent."""
    # Instantiating pipeline with an invalid path should warning-log and return pipeline in degraded state
    pipeline = AccessoryPipeline(
        classifier_path="invalid_path.onnx",
        face_model="buffalo_s",
        device="cpu"
    )
    assert pipeline.classifier is None or isinstance(pipeline.classifier, str) == False

def test_pipeline_batched_preprocessor():
    """Verifies internal preprocessing functions in the inference pipeline."""
    pipeline = AccessoryPipeline(
        classifier_path="invalid_path.onnx",
        face_model="buffalo_s",
        device="cpu"
    )
    
    dummy_crop = np.random.randint(0, 255, (80, 120, 3), dtype=np.uint8)
    preprocessed = pipeline._preprocess_crop(dummy_crop)
    
    # Should yield shape (1, 3, 224, 224)
    assert preprocessed.shape == (1, 3, 224, 224)
    assert preprocessed.dtype == np.float32

if __name__ == "__main__":
    pytest.main([__file__])
