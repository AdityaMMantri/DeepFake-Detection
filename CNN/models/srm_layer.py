import torch
import torch.nn as nn


class SRMLayer(nn.Module):
    """
    SRM high-pass filter bank for forensic residual extraction.

    - Uses multiple fixed filters
    - Applied independently to each RGB channel
    - Output channels = num_filters * 3
    """

    def __init__(self):
        super().__init__()

        # Define SRM kernels (3x3)
        kernels = torch.tensor([
            [[0,0,0],[0,-1,1],[0,0,0]],
            [[0,0,0],[0,1,-1],[0,0,0]],
            [[0,1,0],[1,-4,1],[0,1,0]],
            [[-1,2,-1],[2,-4,2],[-1,2,-1]],
            [[1,-2,1],[-2,4,-2],[1,-2,1]],
            [[0,-1,0],[-1,4,-1],[0,-1,0]],
            [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],
            [[1,0,-1],[0,0,0],[-1,0,1]],
            [[0,1,0],[1,-2,1],[0,1,0]]
        ], dtype=torch.float32)

        num_filters = kernels.shape[0]

        # shape → (filters, 1, 3, 3)
        kernels = kernels.unsqueeze(1)

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=num_filters * 3,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=3
        )

        # repeat filters for each RGB channel
        weight = kernels.repeat(3,1,1,1)

        self.conv.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):

        return self.conv(x)
    
# Summary:
# 1. Defines a bank of SRM high-pass filters used to extract image noise/residual artifacts.
# 2. Each filter detects small texture or manipulation inconsistencies.
# 3. Filters are applied independently to R, G, and B channels using grouped convolution.
# 4. 9 filters × 3 channels produce 27 residual feature maps.
# 5. Filters are fixed (not trainable) because SRM filters are designed forensic operators.