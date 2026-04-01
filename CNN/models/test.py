from models.deepfake_model import DeepfakeModel
import torch

model = DeepfakeModel()

rgb = torch.randn(4,3,256,256)
fft = torch.randn(4,3,256,256)

out = model(rgb, fft)

print(out.shape)