import sys
import os

# =========================
# FIX IMPORT PATH
# =========================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from utils import config
from dataset.vit_dataset import get_dataloaders
from models.vit_model import build_model
from utils.utils import load_checkpoint


# =========================
# EVALUATION LOOP
# =========================
def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


# =========================
# CONFUSION MATRIX
# =========================
def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()

    classes = ["Real", "Fake"]
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black"
            )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================
# MAIN
# =========================
def main():

    device = config.DEVICE
    print(f"Using device: {device}")

    # =========================
    # Load Model
    # =========================
    model = build_model()

    # 🔥 FIXED PATH (robust)
    checkpoint_path = os.path.join(
        ROOT, "outputs", "checkpoints", "best_model.pth"
    )

    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"❌ Checkpoint not found: {checkpoint_path}")

    load_checkpoint(model, checkpoint_path, device=device)
    model.eval()

    print("✔ Model loaded")

    # =========================
    # Load Test Data
    # =========================
    dataloaders = get_dataloaders()
    test_loader = dataloaders["test"]

    # =========================
    # Evaluate
    # =========================
    preds, labels = evaluate(model, test_loader, device)

    # =========================
    # Metrics
    # =========================
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # =========================
    # Confusion Matrix
    # =========================
    cm = confusion_matrix(labels, preds)

    results_dir = os.path.join(ROOT, "outputs", "results")
    os.makedirs(results_dir, exist_ok=True)

    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, cm_path)

    print(f"\n✔ Confusion matrix saved at: {cm_path}")

    # =========================
    # Save Predictions
    # =========================
    np.save(os.path.join(results_dir, "preds.npy"), preds)
    np.save(os.path.join(results_dir, "labels.npy"), labels)

    print("✔ Predictions saved")


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    main()