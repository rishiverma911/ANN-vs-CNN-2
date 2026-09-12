"""Plots used in the dataset review and model comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from project_utils.training import TrainingHistory


def _show_or_close() -> None:
    """Show figures in interactive sessions and close them in non-GUI scripts."""
    backend = plt.get_backend().lower()
    if "agg" in backend and "nbagg" not in backend:
        plt.close()
    else:
        plt.show()


def plot_class_distribution(
    class_counts: dict[str, int],
    title: str = "Class Distribution",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot bar chart of images per class."""
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=classes, y=counts, hue=classes, palette="viridis", legend=False)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    _show_or_close()


def plot_sample_images(
    samples: list[tuple[str, Image.Image]],
    title: str = "Sample Images by Class",
    save_path: Optional[str | Path] = None,
) -> None:
    """Display a grid of sample images with class labels."""
    if not samples:
        print("No sample images available to display.")
        return

    num_samples = len(samples)
    cols = min(6, num_samples)
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(-1)

    for ax, (class_name, image) in zip(axes, samples):
        ax.imshow(image)
        ax.set_title(class_name)
        ax.axis("off")

    for ax in axes[len(samples) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    _show_or_close()


def plot_training_curves(
    history: TrainingHistory,
    model_name: str,
    save_dir: Optional[str | Path] = None,
) -> None:
    """Plot training/validation loss and accuracy curves."""
    epochs = [m.epoch for m in history.epochs]
    train_loss = [m.train_loss for m in history.epochs]
    val_loss = [m.val_loss for m in history.epochs]
    train_acc = [m.train_accuracy for m in history.epochs]
    val_acc = [m.val_accuracy for m in history.epochs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_loss, marker="o", label="Val Loss")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, marker="o", label="Train Accuracy")
    axes[1].plot(epochs, val_acc, marker="o", label="Val Accuracy")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        prefix = model_name.lower().replace(" ", "_")
        plt.savefig(save_dir / f"{prefix}_training_curves.png", dpi=150, bbox_inches="tight")

    _show_or_close()


def plot_confusion_matrix(
    confusion_mat: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot and optionally save a confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    _show_or_close()


def plot_predictions(
    images: Iterable[Image.Image],
    y_true: list[str],
    y_pred: list[str],
    title: str = "Model Predictions",
    save_path: Optional[str | Path] = None,
    max_images: int = 8,
) -> None:
    """Visualize predictions with actual and predicted labels."""
    images = list(images)
    y_true = list(y_true)
    y_pred = list(y_pred)

    n = min(len(images), max_images)
    if n == 0:
        print("No prediction images to display.")
        return

    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for idx in range(n):
        ax = axes[idx]
        ax.imshow(images[idx])
        color = "green" if y_true[idx] == y_pred[idx] else "red"
        ax.set_title(
            f"Actual: {y_true[idx]}\nPredicted: {y_pred[idx]}",
            color=color,
            fontsize=10,
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    _show_or_close()


def save_comparison_table(
    rows: list[dict],
    save_path: str | Path,
) -> pd.DataFrame:
    """Save model comparison metrics to CSV and return DataFrame."""
    df = pd.DataFrame(rows)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    return df


def print_comparison_table(df: pd.DataFrame) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 72)
    print(f"{'Metric':<22} {'ANN':>20} {'CNN':>20}")
    print("-" * 72)

    if "Model" not in df.columns:
        print(df.to_string(index=False))
        print("=" * 72)
        return

    ann_row = df[df["Model"].str.upper() == "ANN"].iloc[0]
    cnn_row = df[df["Model"].str.upper() == "CNN"].iloc[0]

    float_metric_cols = {
        "Best Validation Accuracy",
        "Test Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1",
        "Precision",
        "Recall",
        "F1 Score",
    }
    metrics = [
        ("Trainable Parameters", "Trainable Parameters"),
        ("Training Time (sec)", "Training Time (seconds)"),
        ("Best Validation Accuracy", "Best Validation Accuracy"),
        ("Evaluation Images", "Evaluation Images"),
        ("Test Accuracy", "Test Accuracy"),
        ("Macro Precision", "Macro Precision"),
        ("Macro Recall", "Macro Recall"),
        ("Macro F1", "Macro F1"),
        ("Weighted F1", "Weighted F1"),
    ]

    for label, col in metrics:
        ann_val = ann_row.get(col, "N/A")
        cnn_val = cnn_row.get(col, "N/A")
        if isinstance(ann_val, float):
            ann_str = (
                f"{ann_val:.4f}"
                if "Accuracy" in col or col in float_metric_cols
                else f"{ann_val:.2f}"
            )
            cnn_str = (
                f"{cnn_val:.4f}"
                if "Accuracy" in col or col in float_metric_cols
                else f"{cnn_val:.2f}"
            )
        else:
            ann_str = str(ann_val)
            cnn_str = str(cnn_val)
        print(f"{label:<22} {ann_str:>20} {cnn_str:>20}")

    print("=" * 72)
