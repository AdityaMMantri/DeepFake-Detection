"""
RGB ViT Branch — ViT-Small/Patch16 for deepfake detection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm
from typing import Tuple, Dict, List, Optional

from src.config import Config


class RGBViTBranch(nn.Module):
    """
    Vision Transformer branch for RGB face analysis.
    """
    
    def __init__(self, cfg: Config):
        super().__init__()
        
        self.embed_dim = cfg.vit_embed_dim
        self.input_size = cfg.image_size
        
        # Load ViT-Small/16 from timm
        self.vit = timm.create_model(
            cfg.vit_model_name,
            pretrained=cfg.vit_pretrained,
            num_classes=0,
            drop_rate=0.1,
            drop_path_rate=0.1,
        )
        
        # Classification head
        self.classifier = nn.Linear(cfg.vit_embed_dim, 1)
        self.dropout = nn.Dropout(0.2)
        
        # Initialize classifier
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb: (B, 3, 224, 224) — ImageNet-normalized face
        Returns:
            cls_embedding: (B, 384)
            probability: (B, 1)
        """
        cls_embedding = self.vit(rgb)
        cls_embedding = self.dropout(cls_embedding)
        logit = self.classifier(cls_embedding)
        probability = torch.sigmoid(logit)
        return cls_embedding, probability
        
    def get_logit(self, rgb: torch.Tensor) -> torch.Tensor:
        """Returns raw logit for BCEWithLogitsLoss."""
        cls_embedding = self.vit(rgb)
        cls_embedding = self.dropout(cls_embedding)
        return self.classifier(cls_embedding)