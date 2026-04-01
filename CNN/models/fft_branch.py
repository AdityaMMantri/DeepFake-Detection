import torch
import torch.nn as nn
import torchvision.models as models


class FFTBranch(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(512, 512)

    def forward(self, x):

        x = self.features(x)      # (B,512,1,1)
        x = torch.flatten(x, 1)   # (B,512)
        x = self.proj(x)          # (B,512)

        return x


# Summary:
# 1. Uses ResNet34 as the backbone CNN for frequency-domain feature extraction.
# 2. FFT images represent spectral information of the original RGB image.
# 3. ResNet34 processes FFT images to detect frequency artifacts left by deepfakes.
# 4. The classifier layer is removed so the network acts as a feature extractor.
# 5. Global average pooling inside ResNet produces a (B,512,1,1) feature map.
# 6. The feature map is flattened into a vector of size (B,512).
# 7. A linear projection layer outputs a 512-dimensional embedding for multimodal fusion.