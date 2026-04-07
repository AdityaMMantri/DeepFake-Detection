# DeepFake Detection System (CNN + ViT + Multimodal)

## Overview

This project implements a **multi-architecture deepfake detection system** combining:

- Multimodal CNN (RGB + FFT + Noise)
- 9-Channel Vision Transformer (ViT)
- Single-Channel ViT (artifact-focused)

The system detects deepfakes by leveraging:

- Spatial inconsistencies (RGB)
- Frequency-domain artifacts (FFT)
- Forensic residual noise (SRM)
- Global attention mechanisms (Transformers)

---

## Models

### 🔹 1. Multimodal CNN

A **three-branch architecture**:

- RGB Branch (ConvNeXt-Tiny) → spatial features  
- FFT Branch (ResNet34) → frequency artifacts  
- Noise Branch (SRM + ResNet18) → manipulation residuals  

**Fusion:** Gated Fusion Module  
**Classifier:** Fully connected MLP  
```

INPUT IMAGE │ ┌───────────────┐ │ RGB Branch │──► └───────────────┘ │ ┌───────────────┐ │ │ FFT Branch │──►│──► Gated Fusion ──► Classifier ──► Output └───────────────┘ │ ┌───────────────┐ │ │ Noise Branch │──► └───────────────┘
```

---

### 🔹 2. 9-Channel Vision Transformer (ViT)

#### Architecture

```

Input (9 Channels)
│
├── RGB (3 channels)
├── FFT (3 channels)
└── Noise / SRM (3 channels)
│
▼
Patch Embedding (16×16 patches)
│
▼
Linear Projection → Token Embeddings
│
▼
[CLS] Token + Positional Encoding
│
▼
Transformer Encoder Blocks (Multi-head Self Attention + MLP)
│
▼
Global Representation ([CLS] token)
│
▼
Fully Connected Head
│
▼
Binary Output (Real / Fake)

```

#### Key Idea

- Combines multiple modalities into a single transformer input  
- Learns cross-modal relationships globally  
- Strong at detecting subtle inconsistencies  

---

### 🔹 3. Single-Channel ViT

#### Architecture

```

Input (1 Channel)
│
├── Grayscale / FFT / Noise
│
▼
Patch Embedding
│
▼
Tokenization + Positional Encoding
│
▼
Transformer Encoder Layers
│
▼
[CLS] Token Representation
│
▼
Classification Head
│
▼
Binary Output

```

#### Key Idea

- Focuses purely on artifact-level signals  
- Removes RGB bias  
- Lightweight and efficient  

---

## Combined System View

```

INPUT IMAGE
│
├── RGB ───────────────► CNN Branch
├── FFT ───────────────► CNN Branch
├── Noise (SRM) ───────► CNN Branch
│                         │
│                         ▼
│                   Fusion Module
│                         │
│                         ▼
│                     Classifier
│
├──► 9-Channel ViT (RGB + FFT + Noise)
│
└──► Single-Channel ViT (artifact input)

```

---

## Project Structure

```

DeepFake-Detection/
│
├── CNN/
│   ├── dataset/
│   ├── models/
│   ├── training/
│   └── testing/
│
├── VIT/                      # 9-channel ViT
│   ├── dataset/
│   ├── models/
│   ├── training/
│   ├── testing/
│   └── utils/
│
├── VIT-SINGLE/              # Single-channel ViT
│   └── src/
│       ├── config.py
│       ├── data/
│       ├── models/
│       ├── training/
│       └── main.py
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── checkpoints/
│   ├── cnn/
│   ├── vit/
│   └── vit_single/
│
├── outputs/
│   ├── results_cnn/
│   ├── results_vit/
│   └── results_vit_single/
│
└── README.md

```

---

## Dataset Format

```

data/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
├── test/
│   ├── real/
│   └── fake/

````

---

## Installation

```bash
git clone https://github.com/AdityaMMantri/DeepFake-Detection.git
cd DeepFake-Detection
pip install -r requirements.txt
````

---

## Training

### CNN

```bash
python -m CNN.training.train
```

### 9-Channel ViT

```bash
python -m VIT.training.train
```

### Single-Channel ViT

```bash
cd VIT-SINGLE
python src/main.py --mode train
```

---

## Evaluation

### CNN

```bash
python -m CNN.testing.test
```

### ViT

```bash
python -m VIT.testing.test
```

### Single-Channel ViT

```bash
cd VIT-SINGLE
python src/main.py --mode test --checkpoint checkpoints/best_acc.pth
```

---

## Outputs

```
outputs/
├── results_cnn/
├── results_vit/
└── results_vit_single/
```

Includes:

* confusion_matrix.png
* predictions
* evaluation metrics

---

## Pretrained Models

[https://huggingface.co/Aditya11031/deepfake-detector-models](https://huggingface.co/Aditya11031/deepfake-detector-models)

### Placement

```
checkpoints/
├── cnn/best_model.pth
├── vit/best_model.pth
└── vit_single/best_model.pth
```

---

## Key Features

* Multimodal learning (RGB + FFT + Noise)
* Transformer-based global reasoning
* Artifact-focused detection
* Modular pipeline
* Scalable architecture

---

## Limitations

* Sensitive to dataset quality
* Fixed threshold (0.5)
* No video modeling
* Limited cross-dataset validation

---

## Future Work

* CNN + ViT ensemble
* Video deepfake detection
* Threshold optimization
* Deployment (API / web app)
* Real-time inference

---

## Author

Aditya Mantri\
Abeer Chourey\
Janvi Jain\
BTech AI & Data Science

---

