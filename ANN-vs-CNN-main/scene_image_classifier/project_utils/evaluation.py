"""Metrics and predictions for trained classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class EvaluationResults:
    """Container for evaluation metrics and raw predictions."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    classification_report: str
    confusion_matrix: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_probs: Optional[np.ndarray] = None

    @property
    def precision(self) -> float:
        """Macro-averaged precision (alias for notebooks and legacy scripts)."""
        return self.macro_precision

    @property
    def recall(self) -> float:
        """Macro-averaged recall."""
        return self.macro_recall

    @property
    def f1(self) -> float:
        """Macro-averaged F1."""
        return self.macro_f1


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> EvaluationResults:
    """
    Evaluate a trained model on every batch in the dataloader.

    Reports accuracy, macro precision/recall/F1, weighted F1, per-class
    metrics (via sklearn classification_report), and a full confusion matrix.
    """
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    label_ids = list(range(len(class_names)))

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(
        y_true, y_pred, average="macro", labels=label_ids, zero_division=0
    )
    macro_recall = recall_score(
        y_true, y_pred, average="macro", labels=label_ids, zero_division=0
    )
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", labels=label_ids, zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", labels=label_ids, zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=class_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    return EvaluationResults(
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        classification_report=report,
        confusion_matrix=cm,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
    )


def print_evaluation_summary(results: EvaluationResults, model_name: str, split: str) -> None:
    """Print headline metrics and the per-class classification report."""
    print(f"\n=== {model_name} — {split} set ===")
    print(f"Accuracy:         {results.accuracy:.4f}")
    print(f"Macro Precision:  {results.macro_precision:.4f}")
    print(f"Macro Recall:     {results.macro_recall:.4f}")
    print(f"Macro F1:         {results.macro_f1:.4f}")
    print(f"Weighted F1:      {results.weighted_f1:.4f}")
    print("\nPer-class report:")
    print(results.classification_report)
