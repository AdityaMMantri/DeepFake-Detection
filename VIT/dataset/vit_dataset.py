import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from VIT.utils import config
from VIT.dataset.transforms import build_9channel_tensor, get_train_transforms, get_val_transforms


class DeepfakeDataset9Ch(Dataset):
    """
    Custom dataset for 9-channel deepfake detection.

    Expected folder structure:
        root/
        ├── real/
        │   ├── img001.jpg
        │   └── ...
        └── fake/
            ├── img001.jpg
            └── ...

    Each image is loaded as RGB, then FFT and Noise modalities are computed.
    All three are concatenated into a 9-channel tensor.
    """

    def __init__(self, root_dir, split="train", image_size=224):
        """
        Args:
            root_dir: path to split directory (e.g., dataset/train/)
            split: 'train', 'val', or 'test'
            image_size: target image size
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        # Select transforms based on split
        if split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

        # Collect all image paths and labels
        self.samples = []
        self.class_to_idx = {"real": 0, "fake": 1}

        for class_name, label in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: {class_dir} not found, skipping.")
                continue

            for fname in os.listdir(class_dir):
                ext = fname.lower().split(".")[-1]
                if ext in ("jpg", "jpeg", "png", "bmp", "tiff", "webp"):
                    self.samples.append((os.path.join(class_dir, fname), label))

        print(f"[{split.upper()}] Loaded {len(self.samples)} images "
              f"(Real: {sum(1 for _, l in self.samples if l == 0)}, "
              f"Fake: {sum(1 for _, l in self.samples if l == 1)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            tensor_9ch: [9, image_size, image_size] tensor
            label: 0 (real) or 1 (fake)
        """
        img_path, label = self.samples[idx]

        # Load image as RGB
        image = cv2.imread(img_path)
        if image is None:
            # Handle corrupted images by returning a black image
            print(f"Warning: Could not read {img_path}, using black image.")
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Build 9-channel tensor (RGB + FFT + Noise)
        tensor_9ch = build_9channel_tensor(image, self.transform, self.image_size)

        return tensor_9ch, label


def get_dataloaders(batch_size=None, num_workers=None):
    """
    Create train, val, and test DataLoaders.

    Args:
        batch_size: batch size (default from config)
        num_workers: number of workers (default from config)

    Returns:
        dict with 'train', 'val', 'test' DataLoaders
    """
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS

    datasets = {
        "train": DeepfakeDataset9Ch(config.TRAIN_DIR, split="train", image_size=config.IMAGE_SIZE),
        "val": DeepfakeDataset9Ch(config.VAL_DIR, split="val", image_size=config.IMAGE_SIZE),
        "test": DeepfakeDataset9Ch(config.TEST_DIR, split="test", image_size=config.IMAGE_SIZE),
    }

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=config.PIN_MEMORY,
            drop_last=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.PIN_MEMORY,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.PIN_MEMORY,
        ),
    }

    return dataloaders
