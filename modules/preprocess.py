from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def prepare_data(config: dict[str, Any]):
    img_size = tuple(config["img_size"])
    batch_size = int(config["batch_size"])

    aug = config.get("augmentations") or {}

    train_transforms = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(bool(aug.get("horizontal_flip", True))),
            transforms.RandomRotation(float(aug.get("rotation_range", 0))),
            transforms.ColorJitter(brightness=tuple(aug.get("brightness_range", (1.0, 1.0)))),
            transforms.ToTensor(),
        ]
    )

    val_test_transforms = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ]
    )

    full_train = datasets.ImageFolder(config["train_dir"], transform=train_transforms)
    test_dataset = datasets.ImageFolder(config["test_dir"], transform=val_test_transforms)

    val_size = max(1, int(0.2 * len(full_train))) if len(full_train) else 0
    train_size = max(0, len(full_train) - val_size)

    if train_size and val_size:
        train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
    else:
        train_dataset, val_dataset = full_train, full_train

    limit_train = config.get("limit_train")
    if limit_train:
        train_dataset = Subset(train_dataset, range(min(len(train_dataset), int(limit_train))))

    limit_val = config.get("limit_val")
    if limit_val:
        val_dataset = Subset(val_dataset, range(min(len(val_dataset), int(limit_val))))

    limit_test = config.get("limit_test")
    if limit_test:
        test_dataset = Subset(test_dataset, range(min(len(test_dataset), int(limit_test))))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    classes = getattr(full_train, "classes", [])
    return train_loader, val_loader, test_loader, classes
