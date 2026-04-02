import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
#sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from CNN.dataset.dataset_builder import build_dataset
from CNN.dataset.multimodel_dataset import DeepfakeDataset
from CNN.models.deepfake_model import DeepfakeModel


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)

    for rgb, fft, label in loop:
        rgb = rgb.to(device)
        fft = fft.to(device)
        label = label.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        output = model(rgb, fft)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = (torch.sigmoid(output) > 0.5).float()
        correct += (preds == label).sum().item()
        total += label.size(0)

        acc = correct / total
        loop.set_postfix(loss=loss.item(), acc=acc)

    return total_loss / len(loader), correct / total


def validate_epoch(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(loader, desc=f"Val Epoch {epoch}", leave=False)

    with torch.no_grad():
        for rgb, fft, label in loop:
            rgb = rgb.to(device)
            fft = fft.to(device)
            label = label.float().unsqueeze(1).to(device)

            output = model(rgb, fft)
            loss = criterion(output, label)

            total_loss += loss.item()

            preds = (torch.sigmoid(output) > 0.5).float()
            correct += (preds == label).sum().item()
            total += label.size(0)

            acc = correct / total
            loop.set_postfix(loss=loss.item(), acc=acc)

    return total_loss / len(loader), correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    train_paths, train_labels = build_dataset(os.path.join(DATA_DIR, "train"))
    val_paths, val_labels     = build_dataset(os.path.join(DATA_DIR, "val"))

    train_dataset = DeepfakeDataset(train_paths, train_labels, train=True)
    val_dataset   = DeepfakeDataset(val_paths,   val_labels,   train=False)  # ← no augmentation on val

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model
    model = DeepfakeModel(pretrained=True).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    # verbose=True removed — deprecated in PyTorch >= 2.2
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=3,
        factor=0.5
    )

    epochs = 20
    best_acc = 0.0
    os.makedirs(os.path.join(PROJECT_ROOT, "checkpoints"), exist_ok=True)

    for epoch in range(1, epochs + 1):

        prev_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, epoch
        )   

        scheduler.step(val_acc)

        # Manual LR change logging (replaces deprecated verbose=True)
        curr_lr = optimizer.param_groups[0]["lr"]
        if curr_lr < prev_lr:
            print(f"  ↓ LR reduced: {prev_lr:.2e} → {curr_lr:.2e}")

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "cnn")
            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"  ✔ Saved best model (val_acc={best_acc:.4f})")

    print(f"\nTraining complete. Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()