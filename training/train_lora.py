import sys
from pathlib import Path

import torch
from torch.optim import AdamW

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAINING_DIR = PROJECT_ROOT / "training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from dataset import create_dataloaders
from lora_model import load_vit, apply_lora


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 10

EPOCHS = 3

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.01

SAVE_DIR = PROJECT_ROOT / "models" / "best_lora"

# ------------------------------------------------------------
# SAFETY SWITCH
#
# Keep this False while testing the pipeline.
# Change to True ONLY when running on a suitable GPU.
# ------------------------------------------------------------

RUN_TRAINING = False


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# ACCURACY
# ============================================================

def calculate_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=1)

    correct = (predictions == labels).sum().item()

    total = labels.size(0)

    return correct / total


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_number, batch in enumerate(dataloader, start=1):

        pixel_values = batch["pixel_values"].to(device)

        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs.loss

        logits = outputs.logits

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_correct += (
            torch.argmax(logits, dim=1) == labels
        ).sum().item()

        total_samples += batch_size

        if batch_number % 100 == 0:
            print(
                f"  Batch {batch_number} | "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    dataloader,
    device
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:

        pixel_values = batch["pixel_values"].to(device)

        labels = batch["labels"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs.loss

        logits = outputs.logits

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_correct += (
            torch.argmax(logits, dim=1) == labels
        ).sum().item()

        total_samples += batch_size

    average_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_best_model(model, epoch, val_loss, val_accuracy):

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = SAVE_DIR / "best_model.pt"

    checkpoint = {
        "epoch": epoch,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "model_state_dict": model.state_dict(),
        "num_classes": NUM_CLASSES,
    }

    torch.save(
        checkpoint,
        checkpoint_path
    )

    print(
        f"\nBest model saved to:\n"
        f"{checkpoint_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ViT + LoRA TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "WARNING: CUDA is not available."
        )

        print(
            "The current machine is CPU-only."
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    train_loader, val_loader = create_dataloaders()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading ViT model...")

    # Your lora_model.py defines load_vit()
    # without arguments.
    model = load_vit()

    # --------------------------------------------------------
    # LoRA
    # --------------------------------------------------------

    print("\nApplying LoRA...")

    lora_result = apply_lora(model)

    # Your current apply_lora() returns:
    #
    # (model, something)
    #
    # This handles that safely.
    if isinstance(lora_result, tuple):

        model = lora_result[0]

    else:

        model = lora_result

    # --------------------------------------------------------
    # Move model to device
    # --------------------------------------------------------

    model.to(DEVICE)

    # --------------------------------------------------------
    # Parameter summary
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    trainable_percentage = (
        trainable_parameters / total_parameters
    ) * 100

    print("\n" + "=" * 70)
    print("MODEL PARAMETER SUMMARY")
    print("=" * 70)

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Trainable percentage: "
        f"{trainable_percentage:.4f}%"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    trainable_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print("\nTraining configuration:")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Batch size: {train_loader.batch_size}")

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not RUN_TRAINING:

        print("\n" + "=" * 70)
        print("TRAINING PIPELINE CHECK PASSED")
        print("=" * 70)

        print(
            "\nRUN_TRAINING is currently False."
        )

        print(
            "No training was started."
        )

        if DEVICE.type == "cpu":

            print(
                "\nYour machine is CPU-only."
            )

            print(
                "Use a GPU environment before setting "
                "RUN_TRAINING = True."
            )

        return

    # --------------------------------------------------------
    # Prevent accidental CPU training
    # --------------------------------------------------------

    if DEVICE.type != "cuda":

        raise RuntimeError(
            "\nCUDA is not available.\n"
            "Training is disabled on CPU for this project.\n"
            "Move the training stage to a GPU environment."
        )

    # --------------------------------------------------------
    # Actual training
    # --------------------------------------------------------

    best_val_accuracy = 0.0

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        print("-" * 70)

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=DEVICE
        )

        print(
            f"\nTraining Loss: {train_loss:.4f}"
        )

        print(
            f"Training Accuracy: "
            f"{train_accuracy:.4f}"
        )

        val_loss, val_accuracy = validate(
            model=model,
            dataloader=val_loader,
            device=DEVICE
        )

        print(
            f"Validation Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{val_accuracy:.4f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            save_best_model(
                model=model,
                epoch=epoch,
                val_loss=val_loss,
                val_accuracy=val_accuracy
            )

            print(
                f"New best validation accuracy: "
                f"{best_val_accuracy:.4f}"
            )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()