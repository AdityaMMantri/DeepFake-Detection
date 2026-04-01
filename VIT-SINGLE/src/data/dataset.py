"""
DeepfakeDataset — PyTorch Dataset for binary real/fake classification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

from src.data.augmentation import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DeepfakeDataset(Dataset):
    """Dataset for binary real/fake classification."""
    
    def __init__(
        self,
        real_dir: str,
        fake_dir: str,
        mode: str = "train",
        image_size: int = 224,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        assert mode in ("train", "val", "test"), f"Invalid mode: {mode}"
        
        self.image_size = image_size
        self.mode = mode
        
        if mode == "train":
            self.transform = get_train_transforms(image_size, mean, std)
        else:
            self.transform = get_val_transforms(image_size, mean, std)
            
        self.samples = self._collect_samples(real_dir, fake_dir)
        logger.info(f"[{mode.upper()}] Loaded {len(self.samples)} images")
        
    def _collect_samples(self, real_dir: str, fake_dir: str) -> List[Tuple[Path, int]]:
        samples = []
        
        # Convert string paths to Path objects
        real_path = Path(real_dir)
        fake_path = Path(fake_dir)
        
        # Check if directories exist
        if not real_path.exists():
            raise RuntimeError(f"❌ Real directory not found: {real_dir}")
        if not fake_path.exists():
            raise RuntimeError(f"❌ Fake directory not found: {fake_dir}")
        
        # Real images (label 0)
        real_count = 0
        for path in real_path.iterdir():
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                samples.append((path, 0))
                real_count += 1
                
        # Fake images (label 1)
        fake_count = 0
        for path in fake_path.iterdir():
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                samples.append((path, 1))
                fake_count += 1
                
        if len(samples) == 0:
            raise RuntimeError(
                f"❌ No images found in:\n"
                f"   Real: {real_dir}\n"
                f"   Fake: {fake_dir}\n"
                f"   Supported extensions: {VALID_EXTENSIONS}"
            )
        
        print(f"   Found: {real_count} real, {fake_count} fake images")
        return samples
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, label = self.samples[idx]
        
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return {
                "rgb": torch.zeros(3, self.image_size, self.image_size),
                "label": torch.tensor(label, dtype=torch.long),
                "path": str(path),
            }
            
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_np = np.array(image)
        transformed = self.transform(image=image_np)
        rgb_tensor = transformed["image"]
        
        return {
            "rgb": rgb_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(path),
        }