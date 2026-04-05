# download_dataset.py — Download and prepare the Kaggle finger-counting dataset
#
# Dataset: https://www.kaggle.com/datasets/koryakinp/fingers
#   ~21,600 grayscale images, 6 classes (0–5 fingers), left/right hands
#
# Usage:
#   python training/download_dataset.py
#
# Prerequisites:
#   pip install kaggle
#   Place your kaggle.json API token in ~/.kaggle/ (Linux/Mac) or
#   C:\Users\<you>\.kaggle\ (Windows).
#   Get it from: Kaggle → Settings → API → Create New Token
#
# What it does:
#   1. Downloads the dataset via Kaggle CLI
#   2. Merges left/right hand folders into dataset/0 … dataset/5
#   3. Converts images to 64x64 grayscale (ready for training)

import os
import sys
import shutil
import zipfile
import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(_PROJECT_ROOT, "dataset")
DOWNLOAD_DIR = os.path.join(_PROJECT_ROOT, "dataset_raw")
ZIP_PATH = os.path.join(DOWNLOAD_DIR, "fingers.zip")
IMG_SIZE = 64
NUM_CLASSES = 6


def download_from_kaggle():
    """Download the dataset using the Kaggle CLI."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Check if already downloaded
    if os.path.exists(ZIP_PATH):
        print(f"ZIP already exists at {ZIP_PATH}, skipping download.")
        return True

    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("ERROR: kaggle package not installed.")
        print("  pip install kaggle")
        return False
    except OSError:
        print("ERROR: Kaggle API token not found.")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll to API → Create New Token")
        print("  3. Save kaggle.json to ~/.kaggle/ or C:\\Users\\<you>\\.kaggle\\")
        return False

    print("Downloading dataset from Kaggle...")
    os.system(f'kaggle datasets download -d koryakinp/fingers -p "{DOWNLOAD_DIR}"')

    if not os.path.exists(ZIP_PATH):
        # Kaggle may name it differently
        for f in os.listdir(DOWNLOAD_DIR):
            if f.endswith(".zip"):
                os.rename(os.path.join(DOWNLOAD_DIR, f), ZIP_PATH)
                break

    if not os.path.exists(ZIP_PATH):
        print("ERROR: Download failed. Try downloading manually from:")
        print("  https://www.kaggle.com/datasets/koryakinp/fingers")
        print(f"  Place the ZIP file at: {ZIP_PATH}")
        return False

    print("Download complete!")
    return True


def extract_zip():
    """Extract the downloaded ZIP file."""
    extract_dir = os.path.join(DOWNLOAD_DIR, "extracted")
    if os.path.exists(extract_dir):
        print("Already extracted, skipping.")
        return extract_dir

    print(f"Extracting {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(extract_dir)
    print("Extraction complete!")
    return extract_dir


def collect_image_files(extract_dir):
    """Find all image files and group by finger count (parsed from filename).

    Filenames look like: uuid_3R.png or uuid_0L.png
    The digit before L/R is the finger count.
    """
    import re
    pattern = re.compile(r"_(\d)([LR])\.png$", re.IGNORECASE)

    class_files = {i: [] for i in range(NUM_CLASSES)}

    for root, dirs, files in os.walk(extract_dir):
        for fname in files:
            m = pattern.search(fname)
            if m:
                finger_count = int(m.group(1))
                if 0 <= finger_count < NUM_CLASSES:
                    class_files[finger_count].append(os.path.join(root, fname))

    return class_files


def prepare_dataset(extract_dir):
    """Parse filenames for class labels, resize to 64x64 grayscale, save to dataset/."""
    print("\nPreparing dataset...")

    class_files = collect_image_files(extract_dir)

    total_found = sum(len(v) for v in class_files.values())
    if total_found == 0:
        print("ERROR: Could not find any labeled images in extracted data.")
        print("Check the extracted files at:")
        print(f"  {extract_dir}")
        return False

    # Create output folders
    for i in range(NUM_CLASSES):
        os.makedirs(os.path.join(DATASET_DIR, str(i)), exist_ok=True)

    total = 0
    for finger_count in range(NUM_CLASSES):
        files = class_files[finger_count]
        out_dir = os.path.join(DATASET_DIR, str(finger_count))
        count = 0

        for fpath in files:
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            out_path = os.path.join(out_dir, f"{count:05d}.jpg")
            cv2.imwrite(out_path, img)
            count += 1

        total += count
        print(f"  Class {finger_count} ({finger_count} fingers): {count} images")

    print(f"\nTotal: {total} images saved to {DATASET_DIR}/")
    return True


def cleanup(extract_dir):
    """Remove raw download files to save space."""
    print(f"\nCleaning up {DOWNLOAD_DIR}...")
    shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
    print("Cleanup done.")


def main():
    print("=" * 50)
    print("  Finger Counting Dataset Downloader")
    print("=" * 50)
    print()

    # Check if dataset already exists
    existing = 0
    for i in range(NUM_CLASSES):
        class_dir = os.path.join(DATASET_DIR, str(i))
        if os.path.exists(class_dir):
            existing += len(os.listdir(class_dir))

    if existing > 100:
        print(f"Dataset already exists with {existing} images.")
        # In non-interactive mode, check for --force flag
        if "--force" not in sys.argv:
            try:
                resp = input("Re-download and overwrite? (y/N): ").strip().lower()
                if resp != "y":
                    print("Keeping existing dataset. Done.")
                    return
            except EOFError:
                print("Non-interactive mode. Use --force to overwrite.")
                return

        # Clear existing dataset
        shutil.rmtree(DATASET_DIR, ignore_errors=True)

    # Step 1: Download
    if not download_from_kaggle():
        print("\n--- Manual download instructions ---")
        print("1. Go to https://www.kaggle.com/datasets/koryakinp/fingers")
        print("2. Click 'Download' (you need a free Kaggle account)")
        print(f"3. Save the ZIP as: {ZIP_PATH}")
        print("4. Run this script again")
        return

    # Step 2: Extract
    extract_dir = extract_zip()

    # Step 3: Prepare (merge L/R, resize, convert)
    if not prepare_dataset(extract_dir):
        return

    # Step 4: Cleanup raw files
    cleanup(extract_dir)

    print("\n" + "=" * 50)
    print("  Dataset ready! Now run:")
    print("    python training/train_model.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
