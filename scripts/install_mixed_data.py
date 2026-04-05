import shutil
from pathlib import Path
from PIL import Image
import random
import hashlib
import zipfile
import subprocess
import logging
from tqdm import tqdm
import argparse

random.seed(42)

RAW_DIR = Path("data/raw")
TEMP_DIR = Path("temp")
TARGET_PER_SOURCE = 500
MANIFEST_PATH = Path('data/manifest.csv')
QUARANTINE_DIR = Path("data/quarantine")  # store corrupt files

def create_folders():
    """
    Create required folders:
    - data/raw
    - temp
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents = True, exist_ok = True)

def collect_images_from_folder(folder_path):
    """
    Walk through all subfolders
    Filter only image files
    Return list of image paths
    """
    root_dir = Path(folder_path)

    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(root_dir.rglob(ext))
    return image_paths


def is_valid_image(path):
    """
    Validate image using PIL
    Return True / False
    """
    try:
        with Image.open(path) as im:
            im.verify()
        
        return True

    except Exception as e:
        print(f"An unexpected error occured for {path}: {e}")
        return False

def create_image_hash(path):
    """
    Compute hash of image for deduplication
    """
    hasher = hashlib.md5()
    
    with open(path, 'rb') as f:
        hasher.update(f.read())
    
    return hasher.hexdigest()

def already_processed(source_prefix):
    return len(list(RAW_DIR.glob(f"{source_prefix}_*"))) >= TARGET_PER_SOURCE  # resume logic


def sample_and_store_images(image_paths, target_count, source_prefix):
    """
    Validate images
    Deduplicate
    Randomly sample target_count
    Rename safely
    Copy to data/raw
    """
    seen_hash = set()
    valid_images = []

    for image_path in tqdm(image_paths, desc = f'Validating {source_prefix}'):
        if not is_valid_image(image_path):
            shutil.copy(image_path, QUARANTINE_DIR / image_path.name)  # quarantine corrupt
            continue
        
        img_hash = create_image_hash(image_path)
        if img_hash in seen_hash:
            continue

        seen_hash.add(img_hash)
        valid_images.append(image_path)

    if len(valid_images) < target_count:
            raise ValueError("Not enough valid unique images!")
        
    sampled_images = random.sample(valid_images, target_count)

    for idx, image_path in enumerate(tqdm(sampled_images, desc=f"Storing {source_prefix}")):
        new_name = f"{source_prefix}_{idx:05d}{image_path.suffix}"
        dest_path = RAW_DIR / new_name
        shutil.copy(image_path, dest_path)

        with open(MANIFEST_PATH, "a") as f:
            f.write(f"{new_name},{source_prefix},{image_path},{create_image_hash(image_path)}\n")


def extract_zip(zip_path, extract_to):
    """
    Open zip
    Extract all contents
    """
    extract_to.mkdir(parents = True, exist_ok = True)

    if extract_to.exists() and any(extract_to.iterdir()):  # skip if already extracted
        logging.info(f"Skipping extraction for {zip_path}")
        return

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def process_local_github_zip(zip_path, target_count):
    """
    1. Extract ZIP
    2. Collect images
    3. Validate images
    4. Sample target_count
    5. Rename + copy to data/raw
    """
    extract_dir = TEMP_DIR / "github_extract"
    extract_zip(zip_path, extract_dir)

    images = collect_images_from_folder(extract_dir)
    sample_and_store_images(images, target_count, source_prefix="gh")

def download_kaggle_dataset(dataset_name, extract_to):
    """
    Use Kaggle API or subprocess
    Download zip
    Extract it
    """
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    if extract_to.exists() and any(extract_to.iterdir()):  # skip if already downloaded
        logging.info(f"Skipping Kaggle download for {dataset_name}")
        return

    # Step 1: Download dataset
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_name,
        "-p",
        str(extract_to),
        "--force"
    ]

    subprocess.run(cmd, check=True)

    # Step 2: Find the downloaded zip
    zip_files = list(extract_to.glob("*.zip"))

    if not zip_files:
        raise FileNotFoundError("No ZIP file found after Kaggle download.")

    zip_path = zip_files[0]

    # Step 3: Extract ZIP
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    # Step 4: Delete ZIP (optional but recommended)
    zip_path.unlink()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--github_zip", type=str, required=True)
    parser.add_argument("--kaggle1", type=str, required=True)
    parser.add_argument("--kaggle2", type=str, required=True)
    return parser.parse_args()

def main():
    create_folders()

    # GitHub ZIP source
    process_local_github_zip(
        zip_path=Path(r"C:\Users\PADELL09\Downloads\data.zip"),
        target_count=TARGET_PER_SOURCE
    )

    # Kaggle source 1
    kg1_dir = TEMP_DIR / "kg1"
    download_kaggle_dataset("jigrubhatt/selfieimagedetectiondataset", kg1_dir)
    images_1 = collect_images_from_folder(kg1_dir)
    sample_and_store_images(images_1, TARGET_PER_SOURCE, source_prefix="kg1")

    # Kaggle source 2
    kg2_dir = TEMP_DIR / "kg2"
    download_kaggle_dataset("shamsaddin97/image-captioning-dataset-random-images", kg2_dir)
    images_2 = collect_images_from_folder(kg2_dir)
    sample_and_store_images(images_2, TARGET_PER_SOURCE, source_prefix="kg2")

    logging.info("Dataset creation complete!")

if __name__ == "__main__":
    main()