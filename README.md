# Face Accessory Detection Pipeline

This repository contains the complete implementation of a production-ready, low-latency, two-stage Face Accessory Detection system. The architecture is designed to identify multiple accessories worn on human faces in real time under diverse scenarios (low light, motion blur, non-frontal angles, and occlusions).

---

## Architecture Overview

The system runs in two stages:
1. **Stage 1 (Face Detection)**: RetinaFace (using the default buffalo_s model from InsightFace) locates all faces in the frame. Faces are cropped with a configurable 25% margin padding on all sides. This padding is essential to capture beanies, helmets, and neck-scarves, which lie outside tight face bounding boxes.
2. **Stage 2 (Attribute Classifier)**: A MobileNetV3-Small backbone is fine-tuned with a custom classification head to predict 10 multi-label target classes. Multiple accessories can be worn and detected simultaneously on a single face.

### Target Classes
* glasses_clear
* glasses_tinted
* helmet_bike
* helmet_hard
* cap_hat
* beanie
* scarf_muffler
* face_mask
* face_shield
* balaclava

---

## Technical Specifications for Latency and Accuracy

### 1. Latency Optimization: Crop Batching
Instead of sequentially feeding each detected face crop through the classifier, the pipeline stacks all preprocessed crops from a frame into a single tensor batch. A single forward pass is executed via ONNX Runtime. This reduces Python loop overhead, memory transfer cycles, and API boundary crossings, achieving the latency target of 6ms or less on GPU and 18ms or less on CPU.

### 2. Accuracy Optimization: Imbalance-Aware Loss
Because classes like balaclavas or face shields have fewer positive examples in standard datasets, standard BCE loss would cause the model to underfit rare accessories. We dynamically compute class-frequency-based positive weights (pos_weight) from the training split. This forces the BCE loss function to weight rare class gradients heavier.

### 3. Generalization: Severe Augmentations
The model uses Albumentations to simulate high-motion and low-quality environments:
* Random brightness and contrast (low-light and night frames)
* Perspective transforms and rotations (profile views and head tilts)
* Motion blur (movement in video feeds)
* Coarse dropout (simulating overlapping accessories)
* JPEG compression and noise (low-resolution webcams or CCTV feeds)

### 4. Per-Class Optimal Decision Thresholds
Rather than applying a static 0.5 probability threshold across all classes, the evaluation module sweeps the validation predictions using precision-recall curves to compute class-specific thresholds that maximize the F1-score. These thresholds are exported to a configuration file and imported by the inference pipeline.

---

## Directory Structure

```
.
├── config.yaml                # Hyperparameters, paths, and augmentation rates
├── requirements.txt           # Python library requirements
├── face_accessory_colab.ipynb # Google Colab runner notebook
├── README.md                  # Project documentation
├── AGENTS.md                  # Development pipeline specifications
├── api/
│   └── server.py              # FastAPI server serving the ONNX pipeline
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py        # Automated dataset downloader
│   │   ├── preprocess.py      # RetinaFace cropping and clamping
│   │   ├── annotate.py        # Label compiler, CLIP auto-labeler, and split builder
│   │   └── dataset.py         # PyTorch Dataset and Albumentations transforms
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py      # MobileNetV3-Small PyTorch model definition
│   │   └── pipeline.py        # End-to-end inference pipeline with face batching
│   ├── train.py               # AMP training loop with checkpoint resume logic
│   ├── evaluate.py            # F1 threshold sweeps and metric reporting
│   └── export.py              # ONNX exporter, graph simplifier, and TensorRT engine compiler
└── tests/
    └── test_pipeline.py       # Pipeline and shape testing suite
```

---

## How to Execute the Pipeline in Google Colab

The pipeline is designed to be executed directly from Google Colab to leverage cloud T4 GPUs. Follow these steps:

### Step 1: Clone Repository and Set Workspace
Ensure you mount your Google Drive to save checkpoints and final output models persistently:
```python
from google.colab import drive
drive.mount('/content/drive')

import os
BASE_DIR = '/content/drive/MyDrive/face_accessory_detector'
os.makedirs(BASE_DIR, exist_ok=True)
%cd {BASE_DIR}

# Clone the repository
!git clone https://github.com/Firojpaudel/Face_occlusion.git
%cd Face_occlusion
```

### Step 2: Install Dependencies
```python
!pip install -r requirements.txt
```

### Step 3: Run Data Acquisition
Downloads FiftyOne (OpenImages v7), Roboflow projects, CelebA (via Hugging Face to avoid quota limits), and LFW:
```python
!python src/data/download.py --config config.yaml
```

### Step 4: Run Preprocessing and Dataset Splitting
Extracts face crops, maps annotations, runs CLIP zero-shot classification on unlabeled crops, and creates an 80/10/10 split:
```python
!python src/data/annotate.py --config config.yaml --autolabel
```

### Step 5: Model Training
Trains the classifier using mixed-precision. If the session times out, re-running this cell will automatically resume from the last epoch:
```python
!python src/train.py --config config.yaml
```

### Step 6: Optimal Threshold Search and Evaluation
Calculates mAP on the test split, sweeps thresholds, and outputs classification metrics:
```python
!python src/evaluate.py --config config.yaml --model outputs/checkpoints/best.pth
```

### Step 7: Export ONNX and Compile TensorRT
Converts the best checkpoint into a simplified ONNX graph and compiles it to a TensorRT engine (trtexec):
```python
!python src/export.py --config config.yaml --model outputs/checkpoints/best.pth
```

### Step 8: Visual Testing
Run the visualization cell in the Colab notebook to upload an image, run it through the RetinaFace -> ONNX classifier pipeline, and view bounding boxes with predicted accessory percentages.
