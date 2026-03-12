import json

import torch

from modules.evaluate import evaluate_model
from modules.model import build_model
from modules.preprocess import prepare_data
from modules.train import train_model
from modules.utils import detect_classes, ensure_dirs, plot_training


def main() -> None:
    with open("config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)

    ensure_dirs([config["results_path"], "models", "plot"])

    if not config.get("classes"):
        config["classes"] = detect_classes(config["train_dir"])

    train_loader, val_loader, test_loader, classes = prepare_data(config)
    num_classes = len(classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config, num_classes).to(device)

    history = train_model(model, train_loader, val_loader, config, device=device)
    evaluate_model(model, test_loader, classes, device=device, results_path=config["results_path"])
    plot_training(history, save_path="plot/training_plot.png")


if __name__ == "__main__":
    main()
