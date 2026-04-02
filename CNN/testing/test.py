import os
import sys

MODEL_NAME = "cnn"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "results_CNN")

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from CNN.dataset.dataset_builder import build_dataset
from CNN.dataset.multimodel_dataset import DeepfakeDataset
from CNN.models.deepfake_model import DeepfakeModel


def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc="Testing", leave=True)

    with torch.no_grad():
        for rgb, fft, label in loop:

            rgb = rgb.to(device)
            fft = fft.to(device)
            label = label.to(device)

            outputs = model(rgb, fft)
            probs = torch.sigmoid(outputs)

            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(label.cpu().numpy().flatten())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig(save_path)
    plt.close()


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    test_paths, test_labels = build_dataset(os.path.join(DATA_DIR, "test"))

    test_dataset = DeepfakeDataset(
        test_paths,
        test_labels,
        train=False   # VERY IMPORTANT
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # =========================
    # Load Model
    # =========================
    model = DeepfakeModel(pretrained=False).to(device)

    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "cnn", "best_model.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print("✔ Loaded trained model")


    preds, labels = evaluate(model, test_loader, device)


    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")


    cm = confusion_matrix(labels, preds)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

    plot_confusion_matrix(cm, cm_path)

    print(f"\n✔ Confusion matrix saved at: {cm_path}")
    np.save(os.path.join(OUTPUT_DIR, "preds.npy"), preds)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)

    print("✔ Predictions saved")


if __name__ == "__main__":
    main()