"""
Training script for the 9-Channel ViT Deepfake Detector.

Usage:
    python train.py                    # Full training
    python train.py --dry-run          # Single batch test run
    python train.py --resume           # Resume from last checkpoint
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import time
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from VIT.utils import config
from VIT.utils.utils import set_seed, setup_logger, EarlyStopping, save_checkpoint, load_checkpoint, count_parameters
from VIT.dataset.vit_dataset import get_dataloaders
from VIT.models.vit_model import build_model


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, logger, dry_run=False):
    """
    Train for one epoch.

    Returns:
        tuple: (average loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [TRAIN]", ncols=100)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        if config.USE_MIXED_PRECISION and device.type == "cuda":
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_MAX_NORM)
            optimizer.step()

        # Metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100. * correct / total:.1f}%"
        })

        if dry_run and batch_idx >= 0:
            logger.info(f"[DRY RUN] Train batch completed successfully. Loss: {loss.item():.4f}")
            break

    avg_loss = running_loss / total
    accuracy = 100. * correct / total

    return avg_loss, accuracy


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, logger, dry_run=False):
    """
    Validate the model.

    Returns:
        tuple: (average loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [VAL]  ", ncols=100)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if config.USE_MIXED_PRECISION and device.type == "cuda":
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100. * correct / total:.1f}%"
        })

        if dry_run and batch_idx >= 0:
            logger.info(f"[DRY RUN] Validation batch completed successfully. Loss: {loss.item():.4f}")
            break

    avg_loss = running_loss / total
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """
    Cosine annealing scheduler with linear warmup.

    Args:
        optimizer: the optimizer
        warmup_epochs: number of warmup epochs
        total_epochs: total training epochs
        min_lr: minimum learning rate

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            import math
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return max(min_lr / config.LEARNING_RATE, 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Train 9-Channel ViT Deepfake Detector")
    parser.add_argument("--dry-run", action="store_true", help="Run a single batch to test the pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    # Setup
    set_seed(config.SEED)
    logger = setup_logger("train", config.LOG_DIR, "training.log")

    logger.info("=" * 60)
    logger.info("9-Channel ViT Deepfake Detector — Training")
    logger.info("=" * 60)

    # Config overrides
    epochs = args.epochs or config.NUM_EPOCHS
    batch_size = args.batch_size or config.BATCH_SIZE
    lr = args.lr or config.LEARNING_RATE

    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Mixed precision: {config.USE_MIXED_PRECISION}")
    logger.info(f"Dataset root: {config.DATASET_ROOT}")

    # Data
    logger.info("Loading datasets...")
    dataloaders = get_dataloaders(batch_size=batch_size)

    # Model
    logger.info("Building model...")
    model = build_model()
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.WARMUP_EPOCHS, epochs)
    scaler = GradScaler() if config.USE_MIXED_PRECISION else None

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        mode="max"
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "latest_model.pth")
        if os.path.exists(ckpt_path):
            info = load_checkpoint(model, ckpt_path, optimizer, scheduler, config.DEVICE)
            start_epoch = info["epoch"] + 1
            best_val_acc = info["val_acc"]
            logger.info(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
        else:
            logger.warning(f"No checkpoint found at {ckpt_path}, starting fresh.")

    # Training loop
    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, scaler,
            config.DEVICE, epoch, logger, dry_run=args.dry_run
        )

        # Validate
        val_loss, val_acc = validate(
            model, dataloaders["val"], criterion,
            config.DEVICE, epoch, logger, dry_run=args.dry_run
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # Logging
        logger.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )

        # Save best model
        is_best = early_stopping(val_acc)
        if is_best:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_acc,
                os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            )
            logger.info(f"  ★ New best model saved! Val Acc: {val_acc:.2f}%")

        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_acc,
            os.path.join(config.CHECKPOINT_DIR, "latest_model.pth")
        )

        # Early stopping check
        if early_stopping.should_stop:
            logger.info(f"Early stopping triggered after epoch {epoch+1}")
            break

        # Dry run — exit after 1 epoch
        if args.dry_run:
            logger.info("[DRY RUN] Pipeline test passed! Exiting.")
            return

    total_time = time.time() - start_time
    logger.info(f"Training complete in {total_time/60:.1f} minutes")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Best model saved to: {os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')}")


if __name__ == "__main__":
    main()
