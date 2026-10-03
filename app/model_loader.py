from pathlib import Path

import torch

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "quantized"
    / "vit_lora_int8.pt"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Quantized model not found:\n{MODEL_PATH}"
        )

    package = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = package["model"]

    class_names = package["class_names"]

    model.eval()

    return model, class_names