"""
Prediction / Inference script for the 9-Channel ViT Deepfake Detector.

Runs the full pipeline: ViT → Feature Extraction → Agent → Final Verdict.

Usage:
    python predict.py --image path/to/face.jpg
    python predict.py --dir path/to/folder/
    python predict.py --image face.jpg --checkpoint path/to/model.pth
"""

import os
import sys
import argparse
import cv2
import torch
import numpy as np

from utils import config
from utils.utils import set_seed, load_checkpoint
from dataset.transforms import build_9channel_tensor, get_val_transforms
from models.vit_model import build_model
from utils.feature_extractor import ForensicFeatureExtractor
from utils.agent import ForensicAgent


def predict_single_image(image_path, model, extractor, agent, device):
    """
    Run full prediction pipeline on a single image.

    Args:
        image_path: path to the image file
        model: trained ViT model
        extractor: ForensicFeatureExtractor instance
        agent: ForensicAgent instance
        device: computation device

    Returns:
        dict: full analysis result from the agent
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Build 9-channel tensor
    transform = get_val_transforms(config.IMAGE_SIZE)
    tensor_9ch = build_9channel_tensor(image, transform, config.IMAGE_SIZE)
    tensor_9ch = tensor_9ch.unsqueeze(0).to(device)  # [1, 9, 224, 224]

    # Extract forensic features
    features = extractor.extract(tensor_9ch)

    # Agent analysis
    result = agent.analyze(features)
    result["image_path"] = image_path

    return result


def main():
    parser = argparse.ArgumentParser(description="9-Channel ViT Deepfake Detector — Prediction")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image")
    parser.add_argument("--dir", type=str, default=None, help="Path to directory of images")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()

    if not args.image and not args.dir:
        print("Error: Provide --image or --dir argument.")
        print("Usage: python predict.py --image path/to/face.jpg")
        sys.exit(1)

    set_seed(config.SEED)

    # Load model
    checkpoint_path = args.checkpoint or os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Please train the model first: python train.py")
        sys.exit(1)

    print("Loading model...")
    model = build_model()
    load_checkpoint(model, checkpoint_path, device=config.DEVICE)
    model.eval()

    # Initialize feature extractor and agent
    extractor = ForensicFeatureExtractor(model)
    agent = ForensicAgent()

    # Collect images to process
    image_paths = []

    if args.image:
        if os.path.isfile(args.image):
            image_paths.append(args.image)
        else:
            print(f"Error: File not found: {args.image}")
            sys.exit(1)

    if args.dir:
        if os.path.isdir(args.dir):
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
            for fname in sorted(os.listdir(args.dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_exts:
                    image_paths.append(os.path.join(args.dir, fname))
            print(f"Found {len(image_paths)} images in {args.dir}")
        else:
            print(f"Error: Directory not found: {args.dir}")
            sys.exit(1)

    if not image_paths:
        print("No valid images found.")
        sys.exit(1)

    # Process each image
    results_summary = []

    for img_path in image_paths:
        print(f"\nAnalyzing: {os.path.basename(img_path)}")
        result = predict_single_image(img_path, model, extractor, agent, config.DEVICE)

        if result:
            # Print forensic report
            print(result["explanation"])

            results_summary.append({
                "file": os.path.basename(img_path),
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "p_vit": result["p_vit"],
                "trust": result["trust_score"],
                "p_final": result["p_final"],
            })

    # Print summary table for batch predictions
    if len(results_summary) > 1:
        print(f"\n{'='*80}")
        print(f"  BATCH PREDICTION SUMMARY")
        print(f"{'='*80}")
        print(f"  {'File':<30} {'Verdict':<8} {'Confidence':>10} {'P_vit':>8} {'Trust':>8} {'P_final':>8}")
        print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

        for r in results_summary:
            print(f"  {r['file']:<30} {r['prediction']:<8} {r['confidence']:>9.1f}% "
                  f"{r['p_vit']:>8.4f} {r['trust']:>8.4f} {r['p_final']:>8.4f}")

        # Statistics
        num_fake = sum(1 for r in results_summary if r["prediction"] == "fake")
        num_real = sum(1 for r in results_summary if r["prediction"] == "real")
        avg_trust = np.mean([r["trust"] for r in results_summary])

        print(f"\n  Total: {len(results_summary)} | Real: {num_real} | Fake: {num_fake}")
        print(f"  Average Trust Score: {avg_trust:.4f}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
