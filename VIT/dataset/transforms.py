"""
Transforms and preprocessing for the 9-Channel ViT.
Handles augmentation for training and standard preprocessing for val/test.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms


# ImageNet normalization stats (applied per 3-channel group)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# For 9 channels: replicate ImageNet stats for RGB, FFT, and Noise
MEAN_9CH = IMAGENET_MEAN * 3  # [R, G, B, R, G, B, R, G, B]
STD_9CH = IMAGENET_STD * 3


def compute_fft(image_np):
    """
    Compute the FFT magnitude spectrum of an image.

    Args:
        image_np: numpy array of shape (H, W, 3), dtype uint8, BGR or RGB

    Returns:
        fft_magnitude: numpy array of shape (H, W, 3), dtype uint8
                       Log-scaled magnitude spectrum per channel.
    """
    fft_channels = []
    for c in range(3):
        channel = image_np[:, :, c].astype(np.float32)

        f_transform = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f_transform)

        magnitude = np.abs(f_shift)
        magnitude = np.log1p(magnitude)

        # Normalize to 0-255
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        magnitude = (magnitude * 255).astype(np.uint8)

        fft_channels.append(magnitude)

    return np.stack(fft_channels, axis=-1)


def compute_noise_residual(image_np, kernel_size=5):
    """
    Compute the noise residual using a high-pass filter.
    Noise = Original - GaussianBlurred(Original)

    Args:
        image_np: numpy array of shape (H, W, 3), dtype uint8
        kernel_size: size of Gaussian kernel

    Returns:
        noise: numpy array of shape (H, W, 3), dtype uint8
               Noise residual map.
    """
    blurred = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)

    noise = image_np.astype(np.float32) - blurred.astype(np.float32)

    # Normalize to 0-255 range
    noise = noise - noise.min()
    noise = (noise / (noise.max() + 1e-8)) * 255
    noise = noise.astype(np.uint8)

    return noise


def get_train_transforms(image_size=224):
    """
    Training transforms with augmentation.
    Applied to each 3-channel group (RGB, FFT, Noise) individually before concat.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),  # Converts to [0, 1] range
    ])


def get_val_transforms(image_size=224):
    """
    Validation/Test transforms (no augmentation).
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def build_9channel_tensor(image_np, transform, image_size=224):
    """
    Build a 9-channel tensor from a single RGB image.

    Pipeline:
        1. Resize image to target size
        2. Compute FFT magnitude spectrum
        3. Compute noise residual
        4. Apply transforms to each modality
        5. Concatenate → [9, H, W] tensor
        6. Apply normalization

    Args:
        image_np: numpy array (H, W, 3), dtype uint8, RGB format
        transform: torchvision transform for the split
        image_size: target image size

    Returns:
        tensor_9ch: torch.Tensor of shape [9, image_size, image_size]
    """
    image_resized = cv2.resize(image_np, (image_size, image_size))

    rgb = image_resized.copy()
    fft = compute_fft(image_resized)
    noise = compute_noise_residual(image_resized)

    # Apply transforms to each modality → [3, H, W] each
    rgb_tensor = transform(rgb)
    fft_tensor = transform(fft)
    noise_tensor = transform(noise)

    # Concatenate → [9, H, W]
    tensor_9ch = torch.cat([rgb_tensor, fft_tensor, noise_tensor], dim=0)

    # Normalize all 9 channels using ImageNet stats
    normalize = transforms.Normalize(mean=MEAN_9CH, std=STD_9CH)
    tensor_9ch = normalize(tensor_9ch)

    return tensor_9ch
