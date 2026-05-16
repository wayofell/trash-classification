"""
Utility functions: metrics logging, plotting, confusion matrix.

All functions assume the project structure:
    experiments/
        results.csv      <- log of all experiments
        plots/           <- generated figures
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score
)


# ---------------------------------------------------------------------------
# Paths (resolved relative to project root)
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Project root: parent of src/."""
    return Path(__file__).resolve().parent.parent


def get_experiments_dir() -> Path:
    """Ensure experiments directory exists, return it."""
    p = get_project_root() / "experiments"
    p.mkdir(parents=True, exist_ok=True)
    (p / "plots").mkdir(parents=True, exist_ok=True)
    (p / "logs").mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Metrics logging
# ---------------------------------------------------------------------------

EXPERIMENT_COLUMNS = [
    "timestamp", "experiment_name", "model", "epochs", "batch_size",
    "optimizer", "learning_rate", "augmentation", "class_weights",
    "train_acc", "val_acc", "test_acc", "test_macro_f1",
    "train_loss", "val_loss", "test_loss",
    "training_time_sec", "notes",
]


def log_experiment(record: Dict, csv_path: Optional[Path] = None) -> None:
    """Append an experiment record to experiments/results.csv."""
    if csv_path is None:
        csv_path = get_experiments_dir() / "results.csv"

    record = {**record}
    record.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))

    # Fill missing columns with empty string for consistent CSV
    row = {col: record.get(col, "") for col in EXPERIMENT_COLUMNS}

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
        history,
        title: str = "",
        save_path: Optional[Path] = None,
) -> None:
    """Plot loss and accuracy curves side by side."""
    h = history.history
    epochs_ran = range(1, len(h["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs_ran, h["loss"], label="train", linewidth=2)
    if "val_loss" in h:
        axes[0].plot(epochs_ran, h["val_loss"], label="val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss{(': ' + title) if title else ''}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs_ran, h["accuracy"], label="train", linewidth=2)
    if "val_accuracy" in h:
        axes[1].plot(epochs_ran, h["val_accuracy"], label="val", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Accuracy{(': ' + title) if title else ''}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Predictions and evaluation
# ---------------------------------------------------------------------------

def collect_predictions(
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model on the entire dataset and return (y_true, y_pred) as numpy arrays.
    Works with logit outputs (applies argmax).
    """
    y_true_all = []
    y_pred_all = []
    for x_batch, y_batch in dataset:
        logits = model.predict(x_batch, verbose=0)
        y_pred = np.argmax(logits, axis=1)
        y_true_all.append(y_batch.numpy())
        y_pred_all.append(y_pred)
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        title: str = "Confusion matrix",
        normalize: bool = True,
        save_path: Optional[Path] = None,
) -> np.ndarray:
    """Plot confusion matrix as a heatmap. Returns the matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_display = cm
        fmt = "d"
        vmax = None

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_display, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar=True, vmin=0, vmax=vmax, ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    return cm


def print_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
) -> Dict:
    """Print sklearn classification report and return as dict."""
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=3, output_dict=True
    )
    print(classification_report(
        y_true, y_pred, target_names=class_names, digits=3
    ))
    return report