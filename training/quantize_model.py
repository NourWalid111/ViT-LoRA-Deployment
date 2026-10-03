import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from lora_model import (
    load_vit,
    apply_lora,
    CLASS_NAMES,
)


# ============================================================
# PATHS
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_lora"
    / "best_model.pt"
)

QUANTIZED_DIR = (
    PROJECT_ROOT
    / "models"
    / "quantized"
)

QUANTIZED_MODEL_PATH = (
    QUANTIZED_DIR
    / "vit_lora_int8.pt"
)


# ============================================================
# HELPERS
# ============================================================

def get_file_size_mb(path):
    return path.stat().st_size / (1024 * 1024)


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def count_linear_layers(model):
    return sum(
        1
        for module in model.modules()
        if isinstance(module, nn.Linear)
    )


# ============================================================
# LOAD TRAINED LORA MODEL
# ============================================================

def load_trained_model():

    print("=" * 70)
    print("LOADING TRAINED LoRA MODEL")
    print("=" * 70)

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    print(
        f"Checkpoint:\n{CHECKPOINT_PATH}"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    print(
        f"Training epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation accuracy: "
        f"{checkpoint['val_accuracy']:.4f}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint['val_loss']:.4f}"
    )

    # --------------------------------------------------------
    # Reconstruct exact architecture
    # --------------------------------------------------------

    print("\nReconstructing ViT...")

    model = load_vit()

    print("\nApplying LoRA...")

    model, target_layers = apply_lora(
        model,
        rank=16,
    )

    print(
        f"\nLoRA target layers: "
        f"{target_layers}"
    )

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    print("\nLoading trained weights...")

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )

    if missing_keys:

        print("\nWARNING: Missing keys:")

        for key in missing_keys:
            print(f"  {key}")

    if unexpected_keys:

        print("\nWARNING: Unexpected keys:")

        for key in unexpected_keys:
            print(f"  {key}")

    if not missing_keys and not unexpected_keys:

        print(
            "All checkpoint weights loaded successfully."
        )

    model.eval()

    return model


# ============================================================
# QUANTIZATION
# ============================================================

def quantize_model(model):

    print("\n" + "=" * 70)
    print("APPLYING DYNAMIC INT8 QUANTIZATION")
    print("=" * 70)

    print(
        f"Linear layers before quantization: "
        f"{count_linear_layers(model)}"
    )

    # Dynamic INT8 quantization works on CPU
    # and targets Linear layers.
    quantized_model = quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )

    quantized_model.eval()

    print(
        f"Linear layers after quantization: "
        f"{count_linear_layers(quantized_model)}"
    )

    print(
        "Dynamic INT8 quantization completed."
    )

    return quantized_model


# ============================================================
# SAVE
# ============================================================

def save_quantized_model(
    model,
    training_metadata,
):

    QUANTIZED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the complete quantized model.
    #
    # The model contains the architecture + quantized weights.
    deployment_package = {
        "model": model,
        "class_names": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "quantization": "dynamic_int8",
        "source_checkpoint": str(
            CHECKPOINT_PATH
        ),
        "training_epoch": training_metadata["epoch"],
        "validation_accuracy": training_metadata[
            "val_accuracy"
        ],
        "validation_loss": training_metadata[
            "val_loss"
        ],
    }

    torch.save(
        deployment_package,
        QUANTIZED_MODEL_PATH,
    )

    print(
        f"\nQuantized model saved to:\n"
        f"{QUANTIZED_MODEL_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ViT + LoRA INT8 QUANTIZATION")
    print("=" * 70)

    print(
        f"\nPyTorch version: "
        f"{torch.__version__}"
    )

    print(
        "Device: CPU"
    )

    # --------------------------------------------------------
    # Original checkpoint size
    # --------------------------------------------------------

    original_size_mb = get_file_size_mb(
        CHECKPOINT_PATH
    )

    print(
        f"\nOriginal checkpoint size: "
        f"{original_size_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = load_trained_model()

    print(
        f"\nModel parameters: "
        f"{count_parameters(model):,}"
    )

    # --------------------------------------------------------
    # Quantize
    # --------------------------------------------------------

    quantized_model = quantize_model(
        model
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_quantized_model(
        quantized_model,
        checkpoint,
    )

    # --------------------------------------------------------
    # Quantized size
    # --------------------------------------------------------

    quantized_size_mb = get_file_size_mb(
        QUANTIZED_MODEL_PATH
    )

    size_reduction_mb = (
        original_size_mb
        - quantized_size_mb
    )

    size_reduction_percent = (
        size_reduction_mb
        / original_size_mb
        * 100
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("QUANTIZATION RESULTS")
    print("=" * 70)

    print(
        f"Original size:     "
        f"{original_size_mb:.2f} MB"
    )

    print(
        f"Quantized size:    "
        f"{quantized_size_mb:.2f} MB"
    )

    print(
        f"Size reduction:    "
        f"{size_reduction_mb:.2f} MB"
    )

    print(
        f"Reduction:         "
        f"{size_reduction_percent:.2f}%"
    )

    print("=" * 70)

    print(
        "\nQuantization completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()