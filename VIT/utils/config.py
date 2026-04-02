import os
import torch

# ============================================================
# PATHS — Update DATASET_ROOT to point to your dataset
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_ROOT = os.path.join(BASE_DIR, "data")

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR   = os.path.join(DATASET_ROOT, "val")
TEST_DIR  = os.path.join(DATASET_ROOT, "test")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "vit")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs_vit")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results_VIT")

# Create output directories
for d in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# MODEL
# ============================================================
MODEL_NAME = "vit_small_patch16_224"  # ViT-Small/16
IMAGE_SIZE = 224
PATCH_SIZE = 16
IN_CHANNELS = 9       # RGB(3) + FFT(3) + Noise(3)
NUM_CLASSES = 2        # Real / Fake
HIDDEN_DIM = 384       # ViT-Small hidden dimension
NUM_HEADS = 6          # ViT-Small attention heads
NUM_LAYERS = 12        # ViT-Small encoder layers
DROP_RATE = 0.1        # Dropout before classification head
PRETRAINED = True      # Use ImageNet-pretrained weights

# ============================================================
# TRAINING
# ============================================================
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.1
GRADIENT_CLIP_MAX_NORM = 1.0
USE_MIXED_PRECISION = True

# Early stopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4

# Data loading
NUM_WORKERS = 4
PIN_MEMORY = True

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed for reproducibility
SEED = 42

# ============================================================
# AGENT — Trust Score Thresholds
# ============================================================
AGENT_ENTROPY_THRESHOLD_HIGH = 3.5     # Above this = unfocused attention
AGENT_ENTROPY_THRESHOLD_LOW = 1.0      # Below this = very focused
AGENT_MARGIN_THRESHOLD_LOW = 0.1       # Below this = uncertain prediction
AGENT_MARGIN_THRESHOLD_HIGH = 0.5      # Above this = very confident
AGENT_AGREEMENT_THRESHOLD = 0.7        # Modality agreement threshold
AGENT_CLS_NORM_MIN = 0.5              # Minimum expected CLS norm
AGENT_CLS_NORM_MAX = 50.0             # Maximum expected CLS norm

# Class names
CLASS_NAMES = ["real", "fake"]
