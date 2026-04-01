"""
Utility functions for 9-Channel ViT Deepfake Detection.
Includes EarlyStopping, checkpoint management, logging, and reproducibility.
"""

import os
import random
import logging
import numpy as np
import torch


def set_seed(seed=42):
    """Set seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name, log_dir, filename="training.log"):
    """
    Setup a logger that writes to both console and file.

    Args:
        name: logger name
        log_dir: directory for log file
        filename: log filename

    Returns:
        logger: configured logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_format)

    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


class EarlyStopping:
    """
    Early stopping to terminate training when validation metric stops improving.

    Tracks the best validation metric and triggers stopping after `patience`
    epochs without improvement greater than `min_delta`.
    """

    def __init__(self, patience=5, min_delta=1e-4, mode="max"):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy-like metrics, 'min' for loss-like metrics
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, value):
        """
        Check if training should stop.

        Args:
            value: current epoch metric value

        Returns:
            bool: True if this is the best value so far
        """
        if self.best_value is None:
            self.best_value = value
            return True

        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, path):
    """
    Save model checkpoint.

    Args:
        model: the model
        optimizer: optimizer state
        scheduler: scheduler state
        epoch: current epoch
        val_acc: validation accuracy
        path: save path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_acc": val_acc,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, path, optimizer=None, scheduler=None, device="cpu"):
    """
    Load model checkpoint.

    Args:
        model: the model to load weights into
        path: checkpoint path
        optimizer: optional optimizer to restore
        scheduler: optional scheduler to restore
        device: device to load to

    Returns:
        dict: checkpoint info (epoch, val_acc)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "val_acc": checkpoint.get("val_acc", 0.0),
    }


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
