"""Download and verify the Intel Image Classification Kaggle dataset.

The script reads the Kaggle access token from KAGGLE_API_TOKEN and never writes
credentials to disk. If the Kaggle zip is already present, the download is
skipped and the archive/extracted dataset are verified instead.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


DATASET_REF = "puneet6060/intel-image-classification"
DATASET_TITLE = "Intel Image Classification"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_REF}"
EXPECTED_ZIP_BYTES = 363_152_213
EXPECTED_RAW_IMAGES = 24_335
EXPECTED_TRAIN_IMAGES = 14_034
EXPECTED_TEST_IMAGES = 3_000
EXPECTED_PRED_IMAGES = 7_301
MIN_APPROXIMATE_DATASET_IMAGES = 24_000
MAX_APPROXIMATE_DATASET_IMAGES = 26_000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
EXPECTED_CLASSES = ("buildings", "forest", "glacier", "mountain", "sea", "street")

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "intel.zip"
DEFAULT_EXTRACT_ROOT = PROJECT_ROOT / "data" / "raw"


def count_images(root: Path) -> int:
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def class_counts(class_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_name in EXPECTED_CLASSES:
        folder = class_root / class_name
        counts[class_name] = (
            sum(1 for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
            if folder.exists()
            else 0
        )
    return counts


def dataset_summary(extract_root: Path) -> dict[str, object]:
    train_root = extract_root / "seg_train" / "seg_train"
    test_root = extract_root / "seg_test" / "seg_test"
    pred_root = extract_root / "seg_pred" / "seg_pred"
    pred_images = (
        sum(1 for path in pred_root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if pred_root.exists()
        else 0
    )
    return {
        "raw_images": count_images(extract_root),
        "train_counts": class_counts(train_root),
        "test_counts": class_counts(test_root),
        "prediction_images": pred_images,
    }


def download_zip(token: str, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = zip_path.with_suffix(zip_path.suffix + ".part")
    request = urllib.request.Request(
        DOWNLOAD_URL,
        headers={"Authorization": f"Bearer {token}", "User-Agent": "ann-vs-cnn/1.0"},
    )

    try:
        with urllib.request.urlopen(request) as response, temp_path.open("wb") as output:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            next_report = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if total and downloaded >= next_report:
                    percent = downloaded / total * 100
                    print(f"Downloaded {downloaded / 1_000_000:.1f} MB ({percent:.0f}%)")
                    next_report += max(total // 10, 1)
    except urllib.error.HTTPError as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Kaggle download failed with HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Kaggle download failed: {exc.reason}") from exc

    temp_path.replace(zip_path)


def safe_extract(zip_path: Path, extract_root: Path) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    resolved_root = extract_root.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (extract_root / member.filename).resolve()
            if not target.is_relative_to(resolved_root):
                raise RuntimeError(f"Unsafe zip member path blocked: {member.filename}")
        archive.extractall(extract_root)


def verify_archive(zip_path: Path) -> int:
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset archive not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt zip member detected: {bad_member}")
        image_count = sum(
            1
            for name in archive.namelist()
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS
        )
    return image_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--extract-root", type=Path, default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zip_path = args.zip_path
    extract_root = args.extract_root

    print(f"Dataset: {DATASET_TITLE} ({DATASET_REF})")
    print(f"Archive: {zip_path}")
    print(f"Extract root: {extract_root}")

    if args.force_download or not zip_path.exists():
        token = os.environ.get("KAGGLE_API_TOKEN")
        if not token:
            print(
                "KAGGLE_API_TOKEN is required to download the dataset. "
                "Set it for this terminal session and rerun.",
                file=sys.stderr,
            )
            return 2
        download_zip(token, zip_path)
    else:
        print("Archive already exists; skipping download.")

    zip_size = zip_path.stat().st_size
    print(f"Archive size: {zip_size:,} bytes")
    if zip_size != EXPECTED_ZIP_BYTES:
        print(
            f"Warning: expected {EXPECTED_ZIP_BYTES:,} bytes for the reference archive."
        )

    archive_images = verify_archive(zip_path)
    print(f"Archive image entries: {archive_images:,}")

    extracted_images = count_images(extract_root) if extract_root.exists() else 0
    if args.force_extract or extracted_images < MIN_APPROXIMATE_DATASET_IMAGES:
        print("Extracting dataset...")
        safe_extract(zip_path, extract_root)
    else:
        print("Extracted dataset already present; skipping extraction.")

    summary = dataset_summary(extract_root)
    train_total = sum(summary["train_counts"].values())
    test_total = sum(summary["test_counts"].values())
    pred_total = int(summary["prediction_images"])
    raw_total = int(summary["raw_images"])

    print(f"Raw image files: {raw_total:,}")
    print(f"Labeled train images: {train_total:,}")
    print(f"Labeled test images: {test_total:,}")
    print(f"Unlabeled prediction images: {pred_total:,}")
    print("Train class counts:", summary["train_counts"])
    print("Test class counts:", summary["test_counts"])

    if not MIN_APPROXIMATE_DATASET_IMAGES <= raw_total <= MAX_APPROXIMATE_DATASET_IMAGES:
        raise RuntimeError(
            f"Expected approximately 25,000 raw images, found {raw_total:,}."
        )
    if (
        raw_total != EXPECTED_RAW_IMAGES
        or train_total != EXPECTED_TRAIN_IMAGES
        or test_total != EXPECTED_TEST_IMAGES
        or pred_total != EXPECTED_PRED_IMAGES
    ):
        print("Warning: counts differ from the known Kaggle reference copy.")

    print("Dataset download/extraction verification complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
