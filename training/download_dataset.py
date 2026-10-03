from pathlib import Path
import hashlib
import zipfile
import requests


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ZIP_PATH = DATA_DIR / "EuroSAT_RGB.zip"
EXTRACT_DIR = DATA_DIR / "EuroSAT"

DOWNLOAD_URL = (
    "https://zenodo.org/records/7711810/files/"
    "EuroSAT_RGB.zip?download=1"
)

EXPECTED_MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"


# ============================================================
# Helper functions
# ============================================================

def calculate_md5(file_path: Path) -> str:

    md5 = hashlib.md5()

    with open(file_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            md5.update(chunk)

    return md5.hexdigest()


def download_file(url: str, destination: Path):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Downloading EuroSAT RGB...")
    print(f"Destination: {destination}")

    response = requests.get(
        url,
        stream=True,
        timeout=60
    )

    response.raise_for_status()

    total_size = int(
        response.headers.get(
            "content-length",
            0
        )
    )

    downloaded = 0

    with open(destination, "wb") as file:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if not chunk:
                continue

            file.write(chunk)
            downloaded += len(chunk)

            if total_size:

                percentage = (
                    downloaded
                    / total_size
                    * 100
                )

                print(
                    f"\rProgress: {percentage:6.2f}%",
                    end=""
                )

    print("\nDownload complete.")


# ============================================================
# Main
# ============================================================

def main():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    if not ZIP_PATH.exists():

        download_file(
            DOWNLOAD_URL,
            ZIP_PATH
        )

    else:

        print(
            f"Dataset archive already exists:\n"
            f"{ZIP_PATH}"
        )

    # --------------------------------------------------------
    # Verify checksum
    # --------------------------------------------------------

    print("\nChecking MD5...")

    actual_md5 = calculate_md5(
        ZIP_PATH
    )

    print("Expected:", EXPECTED_MD5)
    print("Actual:  ", actual_md5)

    if actual_md5 != EXPECTED_MD5:

        raise RuntimeError(
            "Dataset checksum does not match. "
            "The downloaded ZIP may be corrupted."
        )

    print("MD5 verification: OK")

    # --------------------------------------------------------
    # Verify ZIP
    # --------------------------------------------------------

    if not zipfile.is_zipfile(ZIP_PATH):

        raise RuntimeError(
            "The downloaded file is not a valid ZIP archive."
        )

    print("ZIP verification: OK")

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    if not EXTRACT_DIR.exists():

        print("\nExtracting dataset...")

        EXTRACT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        with zipfile.ZipFile(
            ZIP_PATH,
            "r"
        ) as archive:

            archive.extractall(
                EXTRACT_DIR
            )

        print("Extraction complete.")

    else:

        print(
            "\nDataset is already extracted."
        )

    print("\nDataset location:")
    print(EXTRACT_DIR)


if __name__ == "__main__":
    main()