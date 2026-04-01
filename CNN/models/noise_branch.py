import torch
import torch.nn as nn
import torchvision.models as models

from .srm_layer import SRMLayer


class NoiseBranch(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        self.srm = SRMLayer()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        old_conv = backbone.conv1
        old_weight = old_conv.weight

        backbone.conv1 = nn.Conv2d(
            27,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        with torch.no_grad():
            backbone.conv1.weight = nn.Parameter(
                    old_weight.repeat(1, 9, 1, 1) / 27
            )

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(512, 512)

    def forward(self, x):

        x = self.srm(x)           # (B, 27, H, W)
        x = self.features(x)      # (B, 512, 1, 1)
        x = torch.flatten(x, 1)   # (B, 512)
        x = self.fc(x)            # (B, 512)

        return x


# Summary:
# 1. SRM filters convert RGB images into residual noise maps highlighting manipulation artifacts.
# 2. SRM produces 27 channels (9 filters applied to each RGB channel).
# 3. ResNet18 backbone processes these residual maps to learn noise patterns.
# 4. The first convolution layer is modified to accept 27 input channels.
# 5. Pretrained weights are partially reused to stabilize training.
# 6. Feature maps are flattened into a vector.
# 7. A linear layer outputs a 512-dimensional noise embedding for multimodal fusion.