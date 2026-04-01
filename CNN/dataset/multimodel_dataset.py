import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .fft_utils import compute_fft


class DeepfakeDataset(Dataset):
    """
    Multimodal dataset for deepfake detection.

    Returns
    -------
    rgb   : RGB tensor for spatial branch
    fft   : FFT tensor for frequency branch
    label : binary label (0 = real, 1 = fake)

    Noise residuals are computed inside the model using SRM filters.
    """

    def __init__(self, image_paths, labels, train=True):

        self.image_paths = image_paths
        self.labels = labels
        self.train = train

        # augmentation applied only during training
        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.RandomResizedCrop(
                size=256,
                scale=(0.8, 1.0)
            )
        ])

        # RGB transform (ConvNeXt pretrained normalization)
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # FFT transform (no ImageNet normalization)
        self.fft_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]

        # Read and convert to RGB immediately — never work in BGR
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to fixed size before any processing
        img = cv2.resize(img, (256, 256))  # numpy RGB (H, W, 3)

        # Apply augmentation only during training
        if self.train:
            img_pil = self.augment(img)           # returns PIL image
            img_np  = np.array(img_pil)           # back to numpy RGB (H, W, 3)
        else:
            img_np = img                          # already numpy RGB

        # Compute FFT from the SAME (possibly augmented) image
        fft_img = compute_fft(img_np)             # expects RGB numpy, returns RGB numpy

        # Apply transforms
        # rgb_transform accepts PIL or uint8 numpy in RGB order
        rgb = self.rgb_transform(img_np)
        fft = self.fft_transform(fft_img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return rgb, fft, label