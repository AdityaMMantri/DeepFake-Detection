"""
ViT-Small/16 Model with Modified 9-Channel Patch Embedding.

Uses timm's pretrained ViT-Small and modifies:
1. Patch embedding: Conv2d(3, 384) → Conv2d(9, 384) with smart weight init
2. Classification head: → Linear(384, 2) for Real/Fake
3. Exposes attention weights for feature extraction
"""

import torch
import torch.nn as nn
import timm
from VIT.utils import config


class ViT9Channel(nn.Module):
    """
    Vision Transformer with 9-channel input for deepfake detection.

    Architecture:
        Input: [B, 9, 224, 224]  (RGB + FFT + Noise)
            ↓
        Modified Patch Embedding (Conv2d: 9 → 384, kernel 16×16)
            ↓
        CLS Token + Positional Encoding
            ↓
        12× Transformer Encoder Blocks (6 heads, 384 dim)
            ↓
        CLS Token → Dropout → Linear(384, 2)
            ↓
        Output: [B, 2] (Real/Fake logits)
    """

    def __init__(
        self,
        model_name=config.MODEL_NAME,
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        drop_rate=config.DROP_RATE,
    ):
        super().__init__()

        # Load pretrained ViT-Small (3-channel)
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

        # Store config
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = self.vit.embed_dim  # 384 for ViT-Small

        # Modify patch embedding to accept 9 channels
        self._modify_patch_embedding(in_channels)

        # Storage for attention weights (populated during forward pass)
        self.attention_weights = []
        self._register_attention_hooks()

    def _modify_patch_embedding(self, in_channels):
        """
        Replace the 3-channel patch embedding with a 9-channel one.
        Smart init: replicate pretrained RGB weights for FFT and Noise channels.
        """
        old_proj = self.vit.patch_embed.proj  # Conv2d(3, 384, 16, 16)

        # Create new conv with 9 input channels
        new_proj = nn.Conv2d(
            in_channels,
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
        )

        # Smart weight initialization
        with torch.no_grad():
            # Get pretrained 3-channel weights: shape [384, 3, 16, 16]
            old_weight = old_proj.weight.data.clone()  # [384, 3, 16, 16]

            # Replicate for all 9 channels:
            # Channels 0-2 (RGB): use pretrained weights as-is
            # Channels 3-5 (FFT): copy pretrained RGB weights
            # Channels 6-8 (Noise): copy pretrained RGB weights
            new_weight = old_weight.repeat(1, in_channels // 3, 1, 1)  # [384, 9, 16, 16]

            # Scale down to maintain similar activation magnitude
            new_weight = new_weight / (in_channels // 3)

            new_proj.weight.data = new_weight

            # Copy bias
            if old_proj.bias is not None:
                new_proj.bias.data = old_proj.bias.data.clone()

        self.vit.patch_embed.proj = new_proj

    def _register_attention_hooks(self):
        """Register forward hooks to capture attention weights from all layers."""
        self._hook_handles = []
        for block in self.vit.blocks:
            handle = block.attn.register_forward_hook(self._attention_hook)
            self._hook_handles.append(handle)

    def _attention_hook(self, module, input, output):
        """Hook function to capture attention weights."""
        # For timm ViT, we need to compute attention manually from qkv
        B, N, C = input[0].shape
        qkv = module.qkv(input[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        # Compute attention weights
        scale = (C // module.num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)  # [B, heads, N, N]

        self.attention_weights.append(attn.detach())

    def forward(self, x, return_features=False):
        """
        Forward pass.

        Args:
            x: input tensor [B, 9, 224, 224]
            return_features: if True, also return CLS token and patch embeddings

        Returns:
            logits: [B, num_classes]
            features (optional): dict with 'cls_token' and 'patch_embeddings'
        """
        # Clear previous attention weights
        self.attention_weights = []

        if return_features:
            # Get patch embeddings + CLS token before the head
            features = self.vit.forward_features(x)  # [B, num_patches+1, hidden_dim]

            cls_token = features[:, 0]          # [B, hidden_dim] — CLS token
            patch_embeddings = features[:, 1:]  # [B, num_patches, hidden_dim]

            # Classification
            logits = self.vit.head(self.vit.fc_norm(cls_token))

            return logits, {
                "cls_token": cls_token,
                "patch_embeddings": patch_embeddings,
                "attention_weights": list(self.attention_weights),
            }
        else:
            logits = self.vit(x)
            return logits

    def get_attention_weights(self):
        """Return captured attention weights from the last forward pass."""
        return self.attention_weights


def build_model(device=None):
    """
    Build and return the 9-channel ViT model.

    Returns:
        model: ViT9Channel on the specified device
    """
    device = device or config.DEVICE
    model = ViT9Channel()
    model = model.to(device)

    total, trainable = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print(f"Model: {config.MODEL_NAME} (modified for {config.IN_CHANNELS}ch)")
    print(f"Parameters: {total:,} total, {trainable:,} trainable")
    print(f"Device: {device}")

    return model
