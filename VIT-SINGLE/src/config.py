"""
Configuration for RGB ViT Branch.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    
    BASE = "F:/SEM-6/DL/DEEP-FAKE/DL-Project/data"
    train_real_dir: str = f"{BASE}/train/real"
    train_fake_dir: str = f"{BASE}/train/fake"
    val_real_dir: str = f"{BASE}/val/real"
    val_fake_dir: str = f"{BASE}/val/fake"
    test_real_dir: str = f"{BASE}/test/real"
    test_fake_dir: str = f"{BASE}/test/fake"

    # ── Model ─────────────────────────────────────────────────
    vit_model_name: str = "vit_small_patch16_224"
    vit_embed_dim: int = 384
    vit_pretrained: bool = True
    num_classes: int = 1

    # ── Training - ANTI-OVERFITTING SETTINGS ──────────────────
    lr_backbone: float = 5e-6
    lr_head: float = 5e-5
    weight_decay: float = 1e-3
    epochs: int = 30
    warmup_epochs: int = 3
    early_stopping_patience: int = 7
    grad_clip: float = 0.5
    label_smoothing: float = 0.1

    # ── Checkpoint ────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    resume: bool = False
    resume_path: str = ""

    # ── Device ────────────────────────────────────────────────
    device: str = "cuda"

    # ── Inference ─────────────────────────────────────────────
    threshold: float = 0.5
    
    # ── ImageNet normalization ────────────────────────────────
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    def __post_init__(self):
        """Create checkpoint directory after initialization."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)

