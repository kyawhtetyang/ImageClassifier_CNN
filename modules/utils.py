from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training(history, save_path: str | None = None) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(history.get("train_acc", []), label="train_acc")
    plt.plot(history.get("val_acc", []), label="val_acc")
    plt.plot(history.get("train_loss", []), label="train_loss")
    plt.plot(history.get("val_loss", []), label="val_loss")
    plt.title("Training Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)


def save_json(data, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def ensure_dirs(paths) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def detect_classes(dataset_dir: str):
    return sorted([p.name for p in Path(dataset_dir).iterdir() if p.is_dir()])
