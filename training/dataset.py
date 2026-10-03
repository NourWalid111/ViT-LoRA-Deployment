from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoImageProcessor
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/EuroSAT/EuroSAT_RGB")
MODEL_NAME = "google/vit-base-patch16-224"

BATCH_SIZE = 16
VAL_SPLIT = 0.2
SEED = 42


# EuroSAT classes
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

CLASS_TO_ID = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


# ============================================================
# DATASET
# ============================================================

class EuroSATDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "pixel_values": image,
            "labels": torch.tensor(label, dtype=torch.long)
        }


# ============================================================
# FIND IMAGES
# ============================================================

def collect_samples():

    samples = []

    for class_name in CLASS_NAMES:

        class_dir = DATA_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}"
            )

        for image_path in class_dir.glob("*.jpg"):

            label = CLASS_TO_ID[class_name]

            samples.append(
                (image_path, label)
            )

    return samples


# ============================================================
# TRANSFORMS
# ============================================================

def create_transforms():

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    image_mean = processor.image_mean
    image_std = processor.image_std

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(10),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=image_mean,
            std=image_std
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=image_mean,
            std=image_std
        ),
    ])

    return train_transform, val_transform


# ============================================================
# CREATE DATA LOADERS
# ============================================================

def create_dataloaders():

    samples = collect_samples()

    print(f"Total samples: {len(samples)}")

    # Reproducible shuffle
    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        len(samples),
        generator=generator
    ).tolist()

    val_size = int(len(samples) * VAL_SPLIT)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_samples = [
        samples[i]
        for i in train_indices
    ]

    val_samples = [
        samples[i]
        for i in val_indices
    ]

    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    train_transform, val_transform = create_transforms()

    train_dataset = EuroSATDataset(
        train_samples,
        transform=train_transform
    )

    val_dataset = EuroSATDataset(
        val_samples,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("EUROSAT DATASET TEST")
    print("=" * 60)

    train_loader, val_loader = create_dataloaders()

    batch = next(iter(train_loader))

    print("\nBatch information:")
    print("Pixel values shape:", batch["pixel_values"].shape)
    print("Labels shape:", batch["labels"].shape)

    print("\nFirst labels:")
    print(batch["labels"][:10])

    print("\nDataset test successful!")