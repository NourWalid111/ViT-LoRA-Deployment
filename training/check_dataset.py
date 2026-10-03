from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "EuroSAT"

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
    "SeaLake"
]


def main():

    print("Dataset:", DATASET_ROOT)
    print()

    total_images = 0

    for class_name in CLASS_NAMES:

        matches = list(
            DATASET_ROOT.rglob(class_name)
        )

        if not matches:

            print(
                f"[MISSING] {class_name}"
            )

            continue

        class_dir = matches[0]

        images = list(
            class_dir.glob("*.jpg")
        )

        count = len(images)

        total_images += count

        print(
            f"{class_name:<25} {count:>5}"
        )

    print()
    print("-" * 40)
    print(
        f"Total images: {total_images}"
    )


if __name__ == "__main__":
    main()