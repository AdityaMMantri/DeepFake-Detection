import torch
import torch.nn as nn
import torchvision.models as models

class RGBBranch(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.convnext_tiny(weights=weights)

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj = nn.Linear(768,512)

    def forward(self,x):

        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x,1)

        x = self.proj(x)

        return x
# 1. Loads ConvNeXt-Tiny as the backbone CNN for feature extraction.
# 2. Removes the classifier and keeps only convolutional feature layers.
# 3. Extracts deep visual features from the RGB image.
# 4. Uses global average pooling to convert spatial feature maps into a vector.
# 5. Flattens the tensor into shape (B, 768).
# 6. Projects the feature vector to a 512-dimensional embedding.
# 7. Output embedding can be fused with other branches in a multimodal deepfake detector.