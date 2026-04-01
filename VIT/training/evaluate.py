"""
Evaluation script for the 9-Channel ViT Deepfake Detector.

Evaluates the best saved model on the test set and reports:
- Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Confusion matrix (saved as image)
- Classification report (saved as text)
- Agent trust score statistics

Usage:
    python evaluate.py
    python evaluate.py --checkpoint path/to/model.pth
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from utils import config
from utils.utils import set_seed, setup_logger, load_checkpoint
from dataset.vit_dataset import get_dataloaders
from models.vit_model import build_model
from utils.feature_extractor import ForensicFeatureExtractor
from utils.agent import ForensicAgent


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset.

    Returns:
        dict with labels, predictions, probabilities
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", ncols=100):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if config.USE_MIXED_PRECISION and device.type == "cuda":
                with autocast():
                    logits = model(images)
            else:
                logits = model(images)

            probs = F.softmax(logits, dim=-1)
            _, predicted = logits.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # P(fake)

    return {
        "labels": np.array(all_labels),
        "predictions": np.array(all_preds),
        "probabilities": np.array(all_probs),
    }


def evaluate_agent(model, dataloader, device, num_samples=100):
    """
    Evaluate the agent's trust scoring on a subset of test images.

    Args:
        model: trained ViT model
        dataloader: test dataloader
        device: computation device
        num_samples: number of images to analyze with the agent

    Returns:
        dict with agent statistics
    """
    extractor = ForensicFeatureExtractor(model)
    agent = ForensicAgent()

    trust_scores = []
    correct_with_agent = 0
    correct_without_agent = 0
    total = 0

    dataset = dataloader.dataset
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    print(f"\nRunning agent analysis on {len(indices)} samples...")

    for idx in tqdm(indices, desc="Agent Analysis", ncols=100):
        image_tensor, label = dataset[idx]
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Extract features
        features = extractor.extract(image_tensor)

        # Agent analysis
        result = agent.analyze(features)

        trust_scores.append(result["trust_score"])

        # ViT-only prediction
        vit_pred = 1 if features["vit_fake_probability"] >= 0.5 else 0
        if vit_pred == label:
            correct_without_agent += 1

        # Agent-calibrated prediction
        agent_pred = 1 if result["p_final"] >= 0.5 else 0
        if agent_pred == label:
            correct_with_agent += 1

        total += 1

    return {
        "trust_scores": np.array(trust_scores),
        "accuracy_vit_only": correct_without_agent / total * 100,
        "accuracy_with_agent": correct_with_agent / total * 100,
        "mean_trust": np.mean(trust_scores),
        "std_trust": np.std(trust_scores),
        "min_trust": np.min(trust_scores),
        "max_trust": np.max(trust_scores),
    }


def plot_confusion_matrix(labels, predictions, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, predictions)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix"
    )

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 9-Channel ViT Deepfake Detector")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--agent-samples", type=int, default=100, help="Number of samples for agent evaluation")
    args = parser.parse_args()

    set_seed(config.SEED)
    logger = setup_logger("evaluate", config.LOG_DIR, "evaluation.log")

    logger.info("=" * 60)
    logger.info("9-Channel ViT Deepfake Detector — Evaluation")
    logger.info("=" * 60)

    # Load checkpoint
    checkpoint_path = args.checkpoint or os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path): 
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Please train the model first: python train.py")
        sys.exit(1)

    # Build model and load weights
    model = build_model()
    info = load_checkpoint(model, checkpoint_path, device=config.DEVICE)
    logger.info(f"Loaded checkpoint from epoch {info['epoch']+1}, val acc: {info['val_acc']:.2f}%")

    # Get test dataloader
    dataloaders = get_dataloaders()

    # === Standard Evaluation ===
    logger.info("\nRunning standard evaluation on test set...")
    results = evaluate_model(model, dataloaders["test"], config.DEVICE)

    labels = results["labels"]
    preds = results["predictions"]
    probs = results["probabilities"]

    # Metrics
    acc = accuracy_score(labels, preds) * 100
    prec = precision_score(labels, preds, zero_division=0) * 100
    rec = recall_score(labels, preds, zero_division=0) * 100
    f1 = f1_score(labels, preds, zero_division=0) * 100

    try:
        auc = roc_auc_score(labels, probs) * 100
    except ValueError:
        auc = 0.0

    logger.info(f"\n{'='*40}")
    logger.info(f"  TEST SET RESULTS")
    logger.info(f"{'='*40}")
    logger.info(f"  Accuracy:  {acc:.2f}%")
    logger.info(f"  Precision: {prec:.2f}%")
    logger.info(f"  Recall:    {rec:.2f}%")
    logger.info(f"  F1-Score:  {f1:.2f}%")
    logger.info(f"  AUC-ROC:   {auc:.2f}%")
    logger.info(f"{'='*40}")

    # Save confusion matrix
    plot_confusion_matrix(
        labels, preds,
        os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    )

    # Save classification report
    report = classification_report(labels, preds, target_names=config.CLASS_NAMES)
    report_path = os.path.join(config.RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"9-Channel ViT Deepfake Detection — Classification Report\n")
        f.write(f"{'='*60}\n\n")
        f.write(report)
        f.write(f"\nAccuracy: {acc:.2f}%\n")
        f.write(f"AUC-ROC:  {auc:.2f}%\n")
    logger.info(f"Classification report saved to: {report_path}")

    # === Agent Evaluation ===
    logger.info("\nRunning agent evaluation...")
    agent_results = evaluate_agent(model, dataloaders["test"], config.DEVICE, args.agent_samples)

    logger.info(f"\n{'='*40}")
    logger.info(f"  AGENT TRUST SCORE STATISTICS")
    logger.info(f"{'='*40}")
    logger.info(f"  Accuracy (ViT only):    {agent_results['accuracy_vit_only']:.2f}%")
    logger.info(f"  Accuracy (with Agent):  {agent_results['accuracy_with_agent']:.2f}%")
    logger.info(f"  Mean Trust Score:       {agent_results['mean_trust']:.4f}")
    logger.info(f"  Std Trust Score:        {agent_results['std_trust']:.4f}")
    logger.info(f"  Min Trust Score:        {agent_results['min_trust']:.4f}")
    logger.info(f"  Max Trust Score:        {agent_results['max_trust']:.4f}")
    logger.info(f"{'='*40}")

    # Save agent results
    agent_report_path = os.path.join(config.RESULTS_DIR, "agent_report.txt")
    with open(agent_report_path, "w") as f:
        f.write(f"Agent Trust Score Report\n{'='*40}\n\n")
        f.write(f"Samples analyzed: {args.agent_samples}\n")
        f.write(f"Accuracy (ViT only): {agent_results['accuracy_vit_only']:.2f}%\n")
        f.write(f"Accuracy (with Agent): {agent_results['accuracy_with_agent']:.2f}%\n")
        f.write(f"Mean Trust: {agent_results['mean_trust']:.4f}\n")
        f.write(f"Std Trust: {agent_results['std_trust']:.4f}\n")
    logger.info(f"Agent report saved to: {agent_report_path}")

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    import sys
    main()
