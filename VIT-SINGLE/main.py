"""
Entry point for the RGB ViT deepfake detector.
"""

import sys
import os
sys.path.append(os.path.abspath("src"))

import src.config as config
import src.data.dataset as dataset
import src.data.augmentation as augmentation
import src.models.branch_vit_rgb as branch_vit_rgb
import src.training.trainer as trainer

sys.modules['config'] = config
sys.modules['dataset'] = dataset
sys.modules['augmentation'] = augmentation
sys.modules['branch_vit_rgb'] = branch_vit_rgb
sys.modules['trainer'] = trainer

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import torch
from PIL import Image
import numpy as np

from src.config import Config
from src.training.trainer import ViTTrainer
from src.models.branch_vit_rgb import RGBViTBranch
from src.data.augmentation import get_val_transforms
from src.data.dataset import DeepfakeDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score,precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class PredictionResult:
    """Container for prediction results."""
    
    def __init__(self, path: str, probability: float, threshold: float = 0.5):
        self.path = path
        self.probability = probability
        self.verdict = "FAKE" if probability > threshold else "REAL"
        self.confidence = max(probability, 1 - probability)
        
    def print_report(self):
        print("\n" + "=" * 50)
        print(f"Image: {self.path}")
        print("-" * 50)
        print(f"Probability: {self.probability:.4f}")
        print(f"Verdict: {self.verdict}")
        print(f"Confidence: {self.confidence:.2%}")
        print("=" * 50)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "probability": float(self.probability),
            "verdict": self.verdict,
            "confidence": float(self.confidence)
        }


class InferencePipeline:
    """Pipeline for inference on images."""
    
    def __init__(self, checkpoint_path: str, cfg: Config):
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.threshold = cfg.threshold
        
        # Load model
        self.model = RGBViTBranch(cfg).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        self.transform = get_val_transforms(cfg.image_size, cfg.mean, cfg.std)
        logger.info(f"Model loaded from {checkpoint_path}")
        
    def preprocess(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        transformed = self.transform(image=image_np)
        return transformed["image"].unsqueeze(0).to(self.device)
        
    @torch.no_grad()
    def predict(self, image_path: str) -> PredictionResult:
        tensor = self.preprocess(image_path)
        _, probability = self.model(tensor)
        return PredictionResult(image_path, probability.item(), self.threshold)
        
    @torch.no_grad()
    def predict_batch(self, image_paths: List[str]) -> List[PredictionResult]:
        results = []
        for path in image_paths:
            try:
                results.append(self.predict(path))
                logger.info(f"{path}: {results[-1].verdict}")
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
        return results


def evaluate_on_test_set(cfg: Config, checkpoint_path: str):
    """Evaluate model on test set."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = RGBViTBranch(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Create test dataset
    test_dataset = DeepfakeDataset(
        real_dir=cfg.test_real_dir,
        fake_dir=cfg.test_fake_dir,
        mode="test",
        image_size=cfg.image_size,
        mean=cfg.mean,
        std=cfg.std,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    
    # Run evaluation
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            rgb = batch["rgb"].to(device)
            labels = batch["label"]
            _, probs = model(rgb)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels)
    
    preds = (all_probs > cfg.threshold).astype(int)
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    precision = precision_score(all_labels, preds)
    recall = recall_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, preds)
    
    print("\n" + "=" * 50)
    print("📊 TEST SET RESULTS")
    print("-" * 50)
    print(f"ACCURACY : {acc:.2%}")
    print(f"AUC      : {auc:.4f}")
    print(f"F1 SCORE : {f1:.4f}")
    print(f"PRECISION: {precision:.4f}")
    print(f"RECALL   : {recall:.4f}")
    print("=" * 50)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['REAL', 'FAKE'],
            yticklabels=['REAL', 'FAKE'])

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    save_path = "confusion_matrix.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Confusion matrix saved at: {save_path}")
    
    return {"accuracy": acc, "auc": auc, "f1": f1}


def parse_args():
    parser = argparse.ArgumentParser(description="RGB ViT Deepfake Detector")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["train", "predict", "predict_folder", "test"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_acc.pth")
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--folder", type=str, default="")
    parser.add_argument("--output_json", type=str, default="predictions.json")
    return parser.parse_args()


def main():
    # Print GPU info
    print("\n" + "=" * 60)
    print("SYSTEM CHECK")
    print("-" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60 + "\n")
    
    args = parse_args()
    
    cfg = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device="cuda",
        resume=args.resume,
        resume_path=args.resume_path,
    )
    
    if args.mode == "train":
        trainer = ViTTrainer(cfg)
        trainer.train()
        
    elif args.mode == "test":
        if not args.checkpoint:
            raise ValueError("--checkpoint required")
        evaluate_on_test_set(cfg, args.checkpoint)
        
    elif args.mode == "predict":
        if not args.image or not args.checkpoint:
            raise ValueError("--image and --checkpoint required")
        pipeline = InferencePipeline(args.checkpoint, cfg)
        pipeline.predict(args.image).print_report()
        
    elif args.mode == "predict_folder":
        if not args.folder or not args.checkpoint:
            raise ValueError("--folder and --checkpoint required")
            
        VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = [
            str(p) for p in Path(args.folder).iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXT
        ]
        
        if not image_paths:
            logger.warning(f"No images found in {args.folder}")
            return
            
        logger.info(f"Found {len(image_paths)} images")
        pipeline = InferencePipeline(args.checkpoint, cfg)
        results = pipeline.predict_batch(image_paths)
        
        fake_count = sum(1 for r in results if r.verdict == "FAKE")
        real_count = sum(1 for r in results if r.verdict == "REAL")
        logger.info(f"Results: FAKE={fake_count} | REAL={real_count}")
        
        with open(args.output_json, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()