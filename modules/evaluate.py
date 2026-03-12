from __future__ import annotations

from typing import Any

import torch
from sklearn.metrics import classification_report, confusion_matrix

from modules.utils import save_json


def evaluate_model(model, test_loader, classes, device: str = "cpu", results_path: str = "report"):
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(outputs.argmax(1).cpu().numpy().tolist())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cr = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    save_json({"confusion_matrix": cm.tolist(), "classification_report": cr}, f"{results_path}/report.json")
    print("Confusion Matrix:\n", cm)
    print("Classification Report saved to report.json")
    return cm, cr
