from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, val_loader, config: dict[str, Any], device: str = "cpu"):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    epochs = int(config.get("epochs", 10))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * imgs.size(0)
            correct += int((outputs.argmax(1) == labels).sum().item())
            total += int(imgs.size(0))

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        model.eval()
        val_total_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_total_loss += float(loss.item()) * imgs.size(0)
                val_correct += int((outputs.argmax(1) == labels).sum().item())
                val_total += int(imgs.size(0))

        val_loss = val_total_loss / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config["save_model_path"])

    torch.save(model.state_dict(), config["save_last_model_path"])
    return history
