"""Training loop and experiment bookkeeping."""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Count trainable, non-trainable, and total parameters in a model.

    Returns a dictionary with keys:
        trainable, non_trainable, total
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    non_trainable = total - trainable
    return {
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total": total,
    }


def get_device_info() -> tuple[torch.device, str]:
    """Detect and describe the compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        info = f"CUDA available. GPU: {gpu_name} (count={gpu_count})"
    else:
        device = torch.device("cpu")
        info = "CUDA not available. Using CPU."
    return device, info


@dataclass
class EpochMetrics:
    """Metrics recorded for a single training epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    epoch_time_sec: float


@dataclass
class TrainingHistory:
    """Full training history and timing information."""

    epochs: list[EpochMetrics] = field(default_factory=list)
    total_training_time_sec: float = 0.0
    best_val_accuracy: float = 0.0
    best_val_loss: float = float("inf")
    best_epoch: int = 0


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    """Run one training or validation epoch."""
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / max(total, 1)
    epoch_accuracy = correct / max(total, 1)
    return epoch_loss, epoch_accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    patience: int = 3,
    model_save_path: Optional[str | Path] = None,
    monitor: str = "val_loss",
    verbose: bool = True,
) -> tuple[TrainingHistory, nn.Module]:
    """
    Train a model with optional early stopping.

    Saves the best model checkpoint to model_save_path when provided.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = TrainingHistory()
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device, train=False
        )
        epoch_time = time.time() - epoch_start

        history.epochs.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                epoch_time_sec=epoch_time,
            )
        )

        improved = False
        if monitor == "val_accuracy":
            if val_acc > history.best_val_accuracy:
                history.best_val_accuracy = val_acc
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                improved = True
        else:
            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                history.best_val_accuracy = val_acc
                history.best_epoch = epoch
                improved = True

        if improved:
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            if model_save_path is not None:
                Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, model_save_path)
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"Epoch {epoch:02d}/{num_epochs:02d}\n"
                f"Train Loss: {train_loss:.4f}\n"
                f"Train Accuracy: {train_acc:.4f}\n"
                f"Val Loss: {val_loss:.4f}\n"
                f"Val Accuracy: {val_acc:.4f}\n"
                f"Time: {epoch_time:.2f} sec"
            )

        if patience > 0 and epochs_without_improvement >= patience:
            if verbose:
                print(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(patience={patience})."
                )
            break

    history.total_training_time_sec = time.time() - start_time
    model.load_state_dict(best_state)
    return history, model


def history_to_dict(history: TrainingHistory) -> dict:
    """Serialize training history for JSON storage."""
    return {
        "epochs": [asdict(epoch) for epoch in history.epochs],
        "total_training_time_sec": history.total_training_time_sec,
        "best_val_accuracy": history.best_val_accuracy,
        "best_val_loss": history.best_val_loss,
        "best_epoch": history.best_epoch,
    }


def history_from_dict(payload: dict) -> TrainingHistory:
    """Restore training history from JSON payload."""
    history = TrainingHistory(
        total_training_time_sec=float(payload["total_training_time_sec"]),
        best_val_accuracy=float(payload["best_val_accuracy"]),
        best_val_loss=float(payload.get("best_val_loss", float("inf"))),
        best_epoch=int(payload.get("best_epoch", 0)),
    )
    history.epochs = [EpochMetrics(**epoch) for epoch in payload["epochs"]]
    return history


def save_training_histories(
    histories: dict[str, TrainingHistory],
    path: str | Path,
) -> None:
    """Save ANN/CNN training histories to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: history_to_dict(history) for name, history in histories.items()}
    path.write_text(json.dumps(payload, indent=2))


def load_training_histories(path: str | Path) -> dict[str, TrainingHistory]:
    """Load ANN/CNN training histories from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: history_from_dict(data) for name, data in payload.items()}


def print_model_summary(model: nn.Module, model_name: str) -> dict[str, int]:
    """Print architecture-related summary information."""
    params = count_parameters(model)
    print(f"\n=== {model_name} Summary ===")
    print(f"Architecture:\n{model}")
    if hasattr(model, "input_shape"):
        print(f"Input shape: {tuple(model.input_shape)}")
    if hasattr(model, "output_shape"):
        print(f"Output shape: {tuple(model.output_shape)}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    print(f"Total parameters: {params['total']:,}")
    return params
