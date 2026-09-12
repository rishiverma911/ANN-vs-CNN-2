"""Dataset discovery, loading, and image preprocessing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

EXPECTED_CLASSES = {
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
MIN_DATASET_IMAGES = 24_000
MAX_DATASET_IMAGES = 26_000
# Guardrails against the rejected shortcuts (2K train / 1K test).
MIN_LABELED_TRAIN_IMAGES = 14_000
MIN_LABELED_TEST_IMAGES = 3_000


def is_valid_image(path: str) -> bool:
    """Keep unreadable image files out of the dataset."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, Image.UnidentifiedImageError):
        return False


def count_images(root: Path) -> int:
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_class_root(directory: Path, min_classes: int = 3) -> Optional[Path]:
    """Find a directory whose subfolders look like class labels."""
    directory = Path(directory)
    if not directory.exists():
        return None

    best_match: Optional[Path] = None
    best_score = 0

    for root, dirs, _files in os.walk(directory):
        root_path = Path(root)
        class_dirs = [d for d in dirs if (root_path / d).is_dir()]
        if not class_dirs:
            continue

        score = 0
        for class_name in class_dirs:
            class_path = root_path / class_name
            if any(
                child.suffix.lower() in IMAGE_EXTENSIONS
                for child in class_path.iterdir()
                if child.is_file()
            ):
                score += 1

        if score >= min_classes and score > best_score:
            best_score = score
            best_match = root_path

    return best_match


def discover_dataset_paths(dataset_root: str | Path) -> dict[str, Optional[Path]]:
    """Discover train, test, and prediction directories."""
    dataset_root = Path(dataset_root)
    result = {
        "train": None,
        "test": None,
        "pred": None,
        "root": dataset_root,
    }

    if not dataset_root.exists():
        return result

    split_candidates = {
        "train": ["seg_train", "train", "training"],
        "test": ["seg_test", "test", "testing"],
        "pred": ["seg_pred", "pred", "prediction"],
    }

    for split, names in split_candidates.items():
        for name in names:
            candidate = dataset_root / name
            if not candidate.exists():
                continue
            if split == "pred":
                nested = candidate / name
                result[split] = nested if nested.exists() else candidate
                break
            class_root = find_class_root(candidate)
            if class_root is not None:
                result[split] = class_root
                break

    if result["train"] is None:
        class_root = find_class_root(dataset_root)
        if class_root is not None:
            result["train"] = class_root

    return result


def summarize_dataset_inventory(dataset_root: str | Path) -> dict:
    """
    Summarize the full Intel dataset (~25,000 images).

    Typical Kaggle layout:
      seg_train/seg_train  ~14,034 labeled training images
      seg_test/seg_test    ~3,000 labeled test images
      seg_pred/seg_pred    ~7,301 unlabeled prediction images
    """
    dataset_root = Path(dataset_root)
    paths = discover_dataset_paths(dataset_root)

    inventory = {
        "total_raw_images": count_images(dataset_root),
        "labeled_train_images": 0,
        "labeled_test_images": 0,
        "unlabeled_prediction_images": 0,
        "train_class_counts": {},
        "test_class_counts": {},
    }

    if paths["train"] is not None:
        train_summary = summarize_imagefolder(
            ImageFolder(root=str(paths["train"]), is_valid_file=is_valid_image)
        )
        inventory["labeled_train_images"] = train_summary["num_images"]
        inventory["train_class_counts"] = train_summary["class_counts"]

    if paths["test"] is not None:
        test_summary = summarize_imagefolder(
            ImageFolder(root=str(paths["test"]), is_valid_file=is_valid_image)
        )
        inventory["labeled_test_images"] = test_summary["num_images"]
        inventory["test_class_counts"] = test_summary["class_counts"]

    if paths["pred"] is not None:
        pred_root = paths["pred"]
        inventory["unlabeled_prediction_images"] = sum(
            1
            for path in pred_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    return inventory


def validate_dataset_root(dataset_root: str | Path) -> Path:
    """Require the complete ~25K-image Intel dataset."""
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(
            "Dataset directory not found.\n"
            f"Place the Intel Image Classification dataset in:\n"
            f"  {dataset_root}\n"
            "Run: python download_dataset.py --force-extract"
        )

    image_count = count_images(dataset_root)
    if not MIN_DATASET_IMAGES <= image_count <= MAX_DATASET_IMAGES:
        raise ValueError(
            "The complete Intel Image Classification dataset is required "
            f"(~25,000 images across six scene categories). Found {image_count:,}."
        )
    return dataset_root


def get_transforms(
    image_size: int = 150,
    augment: bool = False,
) -> transforms.Compose:
    """Resize to 150x150, convert to tensor, and normalize."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if augment:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def class_counts_from_loader(loader: DataLoader) -> dict[str, int]:
    """Count images per class for a DataLoader."""
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
        targets = [int(base_dataset.targets[i]) for i in indices]
        classes = base_dataset.classes
    elif isinstance(dataset, ImageFolder):
        targets = [int(t) for t in dataset.targets]
        classes = dataset.classes
    else:
        raise TypeError("class_counts_from_loader expects ImageFolder or Subset.")

    counts: dict[str, int] = {name: 0 for name in classes}
    for target in targets:
        counts[classes[target]] += 1
    return counts


def validate_class_coverage(
    loader: DataLoader,
    expected_classes: set[str] = EXPECTED_CLASSES,
) -> dict[str, int]:
    """Require every expected class in a split."""
    counts = class_counts_from_loader(loader)
    if set(counts) != expected_classes or any(counts[name] == 0 for name in expected_classes):
        raise ValueError(
            "Every split must contain all six expected classes. "
            f"Observed counts: {counts}"
        )
    return counts


def summarize_imagefolder(dataset: ImageFolder) -> dict:
    """Return dataset summary statistics."""
    class_counts: dict[str, int] = {}
    for _path, class_idx in dataset.samples:
        class_name = dataset.classes[class_idx]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    return {
        "num_classes": len(dataset.classes),
        "class_names": list(dataset.classes),
        "num_images": len(dataset),
        "class_counts": class_counts,
    }


def create_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Create a reproducible, stratified train/validation split."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    generator = torch.Generator().manual_seed(seed)
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("Stratified splitting requires a dataset with targets.")

    class_to_indices: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        class_to_indices.setdefault(int(target), []).append(index)

    if len(class_to_indices) < 6:
        raise ValueError(
            "Stratified split requires all six class labels. "
            f"Found targets: {sorted(class_to_indices)}"
        )

    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_indices in class_to_indices.values():
        if len(class_indices) < 2:
            raise ValueError("Each class must have at least two images for a stratified split.")
        shuffled = torch.randperm(len(class_indices), generator=generator).tolist()
        shuffled_indices = [class_indices[index] for index in shuffled]
        val_size = max(1, round(len(shuffled_indices) * val_ratio))
        if val_size >= len(shuffled_indices):
            val_size = len(shuffled_indices) - 1
        val_indices.extend(shuffled_indices[:val_size])
        train_indices.extend(shuffled_indices[val_size:])

    if len(train_indices) + len(val_indices) != len(dataset):
        raise ValueError("Stratified split dropped images; every training image must be used.")

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_dataloaders(
    dataset_root: str | Path,
    image_size: int = 150,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> dict:
    """
    Build train, validation, and test DataLoaders from the Intel dataset.

    Uses every labeled image in seg_train (~14K) with a seeded stratified
    train/validation split, and the full official seg_test (~3K) for evaluation.

    There is no 2,000-image training cap and no first-1,000 test restriction.
    """
    dataset_root = validate_dataset_root(dataset_root)
    paths = discover_dataset_paths(dataset_root)

    if paths["train"] is None:
        raise FileNotFoundError(
            f"Could not find class folders under: {dataset_root}\n"
            "Ensure the Intel dataset is extracted correctly."
        )

    train_transform = get_transforms(image_size=image_size, augment=True)
    eval_transform = get_transforms(image_size=image_size, augment=False)

    full_train_dataset = ImageFolder(
        root=str(paths["train"]),
        transform=train_transform,
        is_valid_file=is_valid_image,
    )
    if len(full_train_dataset) < MIN_LABELED_TRAIN_IMAGES:
        raise ValueError(
            "Training must use the full Intel labeled split "
            f"(at least {MIN_LABELED_TRAIN_IMAGES:,} images). "
            f"Found {len(full_train_dataset):,}."
        )
    class_names = full_train_dataset.classes
    if set(class_names) != EXPECTED_CLASSES:
        raise ValueError(
            "Expected exactly these six classes: "
            f"{', '.join(sorted(EXPECTED_CLASSES))}. Found: {', '.join(class_names)}"
        )

    train_subset, val_subset = create_train_val_split(
        full_train_dataset, val_ratio=val_ratio, seed=seed
    )

    val_dataset = ImageFolder(
        root=str(paths["train"]),
        transform=eval_transform,
        is_valid_file=is_valid_image,
    )
    val_subset = Subset(val_dataset, val_subset.indices)

    loaders = {
        "train": DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "class_names": class_names,
        "paths": paths,
        "inventory": summarize_dataset_inventory(dataset_root),
        "train_summary": summarize_imagefolder(full_train_dataset),
    }
    loaders["train_counts"] = validate_class_coverage(loaders["train"])
    loaders["val_counts"] = validate_class_coverage(loaders["val"])

    if paths["test"] is not None:
        test_dataset = ImageFolder(
            root=str(paths["test"]),
            transform=eval_transform,
            is_valid_file=is_valid_image,
        )
        if set(test_dataset.classes) != EXPECTED_CLASSES:
            raise ValueError(
                "Test split must contain exactly these six classes: "
                f"{', '.join(sorted(EXPECTED_CLASSES))}. "
                f"Found: {', '.join(test_dataset.classes)}"
            )
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if len(test_dataset) < MIN_LABELED_TEST_IMAGES:
            raise ValueError(
                "Evaluation must use the complete official test split "
                f"(at least {MIN_LABELED_TEST_IMAGES:,} images). "
                f"Found {len(test_dataset):,}."
            )
        loaders["test_summary"] = summarize_imagefolder(test_dataset)
        loaders["test_counts"] = validate_class_coverage(loaders["test"])
    else:
        raise FileNotFoundError(
            "Official labeled test split is required. "
            "Do not evaluate on a truncated or missing test folder."
        )

    return loaders


def get_raw_sample_images(
    dataset_root: str | Path,
    num_samples_per_class: int = 2,
) -> list[tuple[str, Path]]:
    """Collect sample image paths for EDA without applying transforms."""
    dataset_root = validate_dataset_root(dataset_root)
    paths = discover_dataset_paths(dataset_root)
    if paths["train"] is None:
        return []

    samples: list[tuple[str, Path]] = []
    for class_dir in sorted(paths["train"].iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        image_files = [
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        for image_path in sorted(image_files)[:num_samples_per_class]:
            samples.append((class_name, image_path))
    return samples


def load_image_for_display(image_path: Path) -> Image.Image:
    """Load a PIL image for visualization."""
    return Image.open(image_path).convert("RGB")
