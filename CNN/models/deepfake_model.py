import torch
import torch.nn as nn

from .rgb_branch import RGBBranch
from .fft_branch import FFTBranch
from .noise_branch import NoiseBranch
from .fusion import GatedFusion


class DeepfakeModel(nn.Module):
    """
    Full multimodal deepfake detection model.

    Inputs:
        rgb → RGB image tensor (B, 3, H, W)
        fft → FFT image tensor (B, 3, H, W)

    Noise branch internally computes SRM residuals from RGB input.

    Output:
        logits (B, 1) — use BCEWithLogitsLoss during training
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # feature extraction branches
        self.rgb_branch = RGBBranch(pretrained=pretrained)
        self.fft_branch = FFTBranch(pretrained=pretrained)
        self.noise_branch = NoiseBranch(pretrained=pretrained)

        # fusion module
        self.fusion = GatedFusion(feature_dim=512)

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, rgb, fft):
        """
        Forward pass.

        rgb : RGB image tensor
        fft : FFT image tensor
        """

        # extract features
        f_rgb = self.rgb_branch(rgb)        # (B,512)
        f_fft = self.fft_branch(fft)        # (B,512)
        f_noise = self.noise_branch(rgb)    # (B,512)

        # fuse modalities
        fused = self.fusion(f_rgb, f_fft, f_noise)   # (B,512)

        # classification logit
        output = self.classifier(fused)               # (B,1)

        return output

    def predict_proba(self, rgb, fft):
        """
        Inference helper.
        Returns probability of fake.
        """

        self.eval()

        with torch.no_grad():
            logits = self.forward(rgb, fft)
            probs = torch.sigmoid(logits)

        return probs


# Usage
#
# model = DeepfakeModel(pretrained=True)
# criterion = nn.BCEWithLogitsLoss()
#
# Training:
#   logits = model(rgb, fft)
#   loss = criterion(logits, labels.float().unsqueeze(1))
#
# Inference:
#   probs = model.predict_proba(rgb, fft)
#
# Architecture:
#
# rgb ──► RGBBranch   (ConvNeXt-Tiny) ──► (B,512) ─┐
# fft ──► FFTBranch   (ResNet34)      ──► (B,512) ──► GatedFusion ──► MLP ──► (B,1)
# rgb ──► NoiseBranch (SRM+ResNet18)  ──► (B,512) ─┘