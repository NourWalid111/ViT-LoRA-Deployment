import sys
from pathlib import Path

import torch
from PIL import Image

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAINING_DIR = PROJECT_ROOT / "training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


# ============================================================
# IMPORTS
# ============================================================

from dataset import create_transforms
from lora_model import CLASS_NAMES


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "quantized"
    / "vit_lora_int8.pt"
)


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 70)
    print("QUANTIZED MODEL INFERENCE TEST")
    print("=" * 70)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Quantized model not found:\n{MODEL_PATH}"
        )

    print(
        f"Loading model:\n{MODEL_PATH}"
    )

    package = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = package["model"]

    class_names = package["class_names"]

    model.eval()

    print("\nModel loaded successfully.")

    print(
        f"Number of classes: "
        f"{len(class_names)}"
    )

    print(
        f"Classes: {class_names}"
    )

    # --------------------------------------------------------
    # Create preprocessing
    # --------------------------------------------------------

    _, val_transform = create_transforms()

    # --------------------------------------------------------
    # Find a real EuroSAT image
    # --------------------------------------------------------

    image_paths = list(
        (
            PROJECT_ROOT
            / "data"
            / "EuroSAT"
            / "EuroSAT_RGB"
        ).glob(
            "*/*.jpg"
        )
    )

    if not image_paths:

        raise FileNotFoundError(
            "No EuroSAT images were found."
        )

    image_path = image_paths[0]

    print(
        f"\nTesting image:\n{image_path}"
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    pixel_values = val_transform(
        image
    )

    # Add batch dimension
    pixel_values = pixel_values.unsqueeze(0)

    print(
        f"Input tensor shape: "
        f"{pixel_values.shape}"
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            pixel_values=pixel_values
        )

    logits = outputs.logits

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    predicted_id = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[
        0,
        predicted_id
    ].item()

    predicted_class = class_names[
        predicted_id
    ]

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INFERENCE RESULT")
    print("=" * 70)

    print(
        f"Predicted class: {predicted_class}"
    )

    print(
        f"Confidence: {confidence:.4f}"
    )

    print(
        f"Confidence percentage: "
        f"{confidence * 100:.2f}%"
    )

    print("=" * 70)

    print(
        "\nQuantized model inference test PASSED."
    )


if __name__ == "__main__":
    main()