"""End-to-end ANN vs CNN experiment on the full Intel dataset (~25K images)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from model_architectures.ann_classifier import ANNClassifier
from model_architectures.cnn_classifier import CNNClassifier
from project_utils.data import build_dataloaders, summarize_dataset_inventory
from project_utils.evaluation import evaluate_model, print_evaluation_summary
from project_utils.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_predictions,
    plot_training_curves,
    print_comparison_table,
    save_comparison_table,
)
from project_utils.training import (
    count_parameters,
    get_device_info,
    print_model_summary,
    save_training_histories,
    set_seed,
    train_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "data" / "raw"
RESULTS_ROOT = PROJECT_ROOT / "results"
MODEL_DIR = RESULTS_ROOT / "models"
FIGURE_DIR = RESULTS_ROOT / "figures"
METRICS_DIR = RESULTS_ROOT / "metrics"
PREDICTION_DIR = RESULTS_ROOT / "predictions"

IMAGE_SIZE = 150
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
VAL_RATIO = 0.2
PATIENCE = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate saved checkpoints only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for directory in [MODEL_DIR, FIGURE_DIR, METRICS_DIR, PREDICTION_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    inventory = summarize_dataset_inventory(DATASET_ROOT)
    print("=== Intel Image Classification dataset ===")
    print(f"Total raw images:              {inventory['total_raw_images']:,}")
    print(f"Labeled training images:       {inventory['labeled_train_images']:,}")
    print(f"Labeled test images:           {inventory['labeled_test_images']:,}")
    print(f"Unlabeled prediction images:   {inventory['unlabeled_prediction_images']:,}")
    print(f"Six categories: buildings, forest, glacier, mountain, sea, street")

    set_seed(RANDOM_SEED)
    device, device_info = get_device_info()
    print(device_info)

    loaders = build_dataloaders(
        DATASET_ROOT,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED,
        num_workers=0,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    class_names = loaders["class_names"]
    eval_loader = test_loader
    eval_split = "test"
    if eval_loader is None or len(eval_loader.dataset) < 3000:
        raise RuntimeError("Complete official test set (>= 3,000 images) is required.")

    print(f"\nTraining images (stratified):  {len(train_loader.dataset):,}")
    print(f"Validation images:               {len(val_loader.dataset):,}")
    print(f"Evaluation images ({eval_split}): {len(eval_loader.dataset):,}")

    (MODEL_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2))

    ann_model = ANNClassifier(
        num_classes=len(class_names),
        image_size=IMAGE_SIZE,
        hidden_sizes=(160, 32),
        dropout=0.5,
    )
    cnn_model = CNNClassifier(
        num_classes=len(class_names),
        image_size=IMAGE_SIZE,
        dropout=0.5,
    )
    ann_params = print_model_summary(ann_model, "ANN")
    cnn_params = print_model_summary(cnn_model, "CNN")

    if args.eval_only:
        ann_model.load_state_dict(
            torch.load(MODEL_DIR / "best_ann_model.pth", map_location=device)
        )
        cnn_model.load_state_dict(
            torch.load(MODEL_DIR / "best_cnn_model.pth", map_location=device)
        )
        ann_model.to(device)
        cnn_model.to(device)
        ann_history = cnn_history = None
    else:
        set_seed(RANDOM_SEED)
        ann_history, ann_model = train_model(
            ann_model,
            train_loader,
            val_loader,
            device,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            patience=PATIENCE,
            model_save_path=MODEL_DIR / "best_ann_model.pth",
        )

        set_seed(RANDOM_SEED)
        cnn_history, cnn_model = train_model(
            cnn_model,
            train_loader,
            val_loader,
            device,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            patience=PATIENCE,
            model_save_path=MODEL_DIR / "best_cnn_model.pth",
        )

        print(f"\nANN training time: {ann_history.total_training_time_sec:.2f} s")
        print(f"CNN training time: {cnn_history.total_training_time_sec:.2f} s")
        print(f"ANN best validation accuracy: {ann_history.best_val_accuracy:.4f}")
        print(f"CNN best validation accuracy: {cnn_history.best_val_accuracy:.4f}")
        save_training_histories(
            {"ann": ann_history, "cnn": cnn_history},
            METRICS_DIR / "training_histories.json",
        )

    ann_results = evaluate_model(ann_model, eval_loader, device, class_names)
    cnn_results = evaluate_model(cnn_model, eval_loader, device, class_names)
    print_evaluation_summary(ann_results, "ANN", eval_split)
    print_evaluation_summary(cnn_results, "CNN", eval_split)

    for split_name, counts in [
        ("train", loaders["train_counts"]),
        ("validation", loaders["val_counts"]),
        ("test", loaders.get("test_counts")),
    ]:
        if counts:
            plot_class_distribution(
                counts,
                f"{split_name.title()} class distribution",
                FIGURE_DIR / f"{split_name}_class_distribution.png",
            )

    if ann_history is not None and cnn_history is not None:
        plot_training_curves(ann_history, "ANN", FIGURE_DIR)
        plot_training_curves(cnn_history, "CNN", FIGURE_DIR)
    plot_confusion_matrix(
        ann_results.confusion_matrix,
        class_names,
        f"ANN confusion matrix ({eval_split})",
        FIGURE_DIR / "ann_confusion_matrix.png",
    )
    plot_confusion_matrix(
        cnn_results.confusion_matrix,
        class_names,
        f"CNN confusion matrix ({eval_split})",
        FIGURE_DIR / "cnn_confusion_matrix.png",
    )
    plt.close("all")

    (METRICS_DIR / "ann_classification_report.txt").write_text(
        ann_results.classification_report
    )
    (METRICS_DIR / "cnn_classification_report.txt").write_text(
        cnn_results.classification_report
    )

    comparison_rows = [
        {
            "Model": "ANN",
            "Trainable Parameters": ann_params["trainable"],
            "Training Time (seconds)": (
                ann_history.total_training_time_sec if ann_history else None
            ),
            "Best Validation Accuracy": (
                ann_history.best_val_accuracy if ann_history else None
            ),
            "Evaluation Images": len(eval_loader.dataset),
            "Test Accuracy": ann_results.accuracy,
            "Macro Precision": ann_results.macro_precision,
            "Macro Recall": ann_results.macro_recall,
            "Macro F1": ann_results.macro_f1,
            "Weighted F1": ann_results.weighted_f1,
        },
        {
            "Model": "CNN",
            "Trainable Parameters": cnn_params["trainable"],
            "Training Time (seconds)": (
                cnn_history.total_training_time_sec if cnn_history else None
            ),
            "Best Validation Accuracy": (
                cnn_history.best_val_accuracy if cnn_history else None
            ),
            "Evaluation Images": len(eval_loader.dataset),
            "Test Accuracy": cnn_results.accuracy,
            "Macro Precision": cnn_results.macro_precision,
            "Macro Recall": cnn_results.macro_recall,
            "Macro F1": cnn_results.macro_f1,
            "Weighted F1": cnn_results.weighted_f1,
        },
    ]
    comparison_df = save_comparison_table(comparison_rows, METRICS_DIR / "model_comparison.csv")
    print("\n", comparison_df.to_string(index=False))
    print_comparison_table(comparison_df)

    images, labels = next(iter(eval_loader))
    with torch.inference_mode():
        predictions = cnn_model(images.to(device)).argmax(dim=1).cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    display_images = [
        (image * std + mean).permute(1, 2, 0).clamp(0, 1).numpy() for image in images[:8]
    ]
    plot_predictions(
        display_images,
        [class_names[label] for label in labels[:8]],
        [class_names[label] for label in predictions[:8]],
        "CNN predictions",
        PREDICTION_DIR / "cnn_predictions.png",
    )
    plt.close("all")

    print(f"\nExperiment complete. Artifacts saved under: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
