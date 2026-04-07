import os
import sys

# =========================================================
# 🔴 FINAL IMPORT FIX
# =========================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

sys.path.insert(0, os.path.abspath("VIT"))
sys.path.insert(0, os.path.abspath("VIT/utils"))
sys.path.insert(0, os.path.abspath("VIT-SINGLE"))

# =========================================================
# IMPORTS
# =========================================================
import torch

from CNN.models.deepfake_model import DeepfakeModel

from VIT.models.vit_model import build_model
from VIT.utils.utils import load_checkpoint
from VIT.utils.feature_extractor import ForensicFeatureExtractor
from VIT.utils.agent import ForensicAgent
from VIT.utils import config as vit_config

from src.models.branch_vit_rgb import RGBViTBranch
from src.data.augmentation import get_val_transforms
from src.config import Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# MODEL PATHS
# =========================================================
MODEL_PATHS = {
    "cnn": r"F:\SEM-6\DL\DEEP-FAKE\DL-Project\checkpoints\cnn\best_model.pth",
    "vit_9ch": r"F:\SEM-6\DL\DEEP-FAKE\DL-Project\checkpoints\vit\best_model.pth",
    "vit_single": r"F:\SEM-6\DL\DEEP-FAKE\DL-Project\checkpoints\vit_single\best_auc.pth"
}


# =========================================================
# CNN
# =========================================================
class CNNSystem:
    def __init__(self, checkpoint_path):
        print("\n========== LOADING CNN ==========")
        print(f"[CNN] Checkpoint: {checkpoint_path}")

        self.name = "cnn"

        self.model = DeepfakeModel(pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

        self.model.to(DEVICE)
        self.model.eval()

        print(f"[CNN] Model Loaded: {self.model.__class__.__name__}")

    def predict(self, image_np):
        print("[CNN] Running inference")

        import cv2
        from torchvision import transforms
        from CNN.dataset.fft_utils import compute_fft

        image_np = cv2.resize(image_np, (256, 256))
        fft_img = compute_fft(image_np)

        rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])(image_np).unsqueeze(0).to(DEVICE)

        fft = transforms.ToTensor()(fft_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(rgb, fft)).item()

        return {"prediction": "fake" if prob > 0.5 else "real",
                "confidence": prob if prob > 0.5 else 1 - prob,
                "p_fake": prob}


# =========================================================
# VIT 9CH
# =========================================================
class ViT9ChSystem:
    def __init__(self, checkpoint_path):
        print("\n========== LOADING VIT 9CH ==========")
        print(f"[VIT-9CH] Checkpoint: {checkpoint_path}")

        self.name = "vit_9ch"

        self.model = build_model()
        load_checkpoint(self.model, checkpoint_path, device=DEVICE)

        self.model.to(DEVICE)
        self.model.eval()

        self.extractor = ForensicFeatureExtractor(self.model)
        self.agent = ForensicAgent()

        print(f"[VIT-9CH] Model Loaded: {self.model.__class__.__name__}")

    def predict(self, image_np):
        print("[VIT-9CH] Running inference")

        from VIT.dataset.transforms import build_9channel_tensor, get_val_transforms

        transform = get_val_transforms(vit_config.IMAGE_SIZE)

        tensor = build_9channel_tensor(image_np, transform, vit_config.IMAGE_SIZE)
        tensor = tensor.unsqueeze(0).to(DEVICE)

        features = self.extractor.extract(tensor)
        return self.agent.analyze(features)


# =========================================================
# VIT SINGLE
# =========================================================
class ViTSingleSystem:
    def __init__(self, checkpoint_path):
        print("\n========== LOADING VIT SINGLE ==========")
        print(f"[VIT-SINGLE] Checkpoint: {checkpoint_path}")

        self.name = "vit_single"

        self.cfg = Config()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")

        self.model = RGBViTBranch(self.cfg).to(self.device)

        # 🔥 FIX: pickle config mapping
        import importlib
        vit_single_config = importlib.import_module("src.config")
        sys.modules["config"] = vit_single_config

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        del sys.modules["config"]

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"[VIT-SINGLE] Model Loaded: {self.model.__class__.__name__}")

        self.transform = get_val_transforms(
            self.cfg.image_size,
            self.cfg.mean,
            self.cfg.std
        )

    def predict(self, image_np):
        print("[VIT-SINGLE] Running inference")

        tensor = self.transform(image=image_np)["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, prob = self.model(tensor)

        prob = prob.item()

        return {"prediction": "fake" if prob > 0.5 else "real",
                "confidence": prob if prob > 0.5 else 1 - prob,
                "p_fake": prob}


# =========================================================
# MAIN LOADER
# =========================================================
def load_model(model_type):
    print(f"\n[INFO] Requested model: {model_type}")

    if model_type == "cnn":
        model = CNNSystem(MODEL_PATHS["cnn"])

    elif model_type == "vit_9ch":
        model = ViT9ChSystem(MODEL_PATHS["vit_9ch"])

    elif model_type == "vit_single":
        model = ViTSingleSystem(MODEL_PATHS["vit_single"])

    else:
        raise ValueError(f"Invalid model type: {model_type}")

    print(f"[CONFIRM] ACTIVE MODEL: {model.name.upper()}")
    return model