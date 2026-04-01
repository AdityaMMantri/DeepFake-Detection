import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.proj = nn.Linear(feature_dim * 3, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, f_rgb, f_fft, f_noise):

        concat = torch.cat([f_rgb, f_fft, f_noise], dim=1)

        weights = torch.softmax(self.gate(concat), dim=1)

        w_rgb   = weights[:,0].unsqueeze(1)
        w_fft   = weights[:,1].unsqueeze(1)
        w_noise = weights[:,2].unsqueeze(1)

        scaled = torch.cat([
            w_rgb * f_rgb,
            w_fft * f_fft,
            w_noise * f_noise
        ], dim=1)

        fused = self.norm(self.proj(scaled))

        return fused