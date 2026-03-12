from __future__ import annotations

from typing import Any

import torch.nn as nn
import torchvision.models as models


def build_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    model_type = (config.get("model_type") or "transfer").lower()

    if model_type == "transfer":
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        fine_tune = bool(config.get("fine_tune", True))
        if not fine_tune:
            for param in base.parameters():
                param.requires_grad = False

        base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(base.last_channel, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        return base

    # Small custom CNN baseline
    img_size = tuple(config.get("img_size", (64, 64)))
    h, w = int(img_size[0]), int(img_size[1])
    flattened = 128 * (h // 8) * (w // 8)

    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(flattened, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, num_classes),
    )
