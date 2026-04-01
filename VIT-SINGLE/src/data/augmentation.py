"""
Augmentation pipelines with stronger regularization.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Optional


def get_train_transforms(
    image_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> A.Compose:
    """Training augmentation pipeline."""
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
        
    return A.Compose([
        A.RandomResizedCrop(
            height=image_size,
            width=image_size,
            scale=(0.7, 1.0),
            ratio=(0.8, 1.2),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
            p=0.7
        ),
        A.ImageCompression(
            quality_lower=50,
            quality_upper=100,
            p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
        A.Rotate(limit=15, border_mode=0, p=0.4),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
        A.CoarseDropout(
            max_holes=8, 
            max_height=16, 
            max_width=16, 
            fill_value=0, 
            p=0.3
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(
    image_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> A.Compose:
    """Validation/inference transform."""
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
        
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])