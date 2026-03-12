import json
import sys

import torch
from PIL import Image
from torchvision import transforms

from modules.model import build_model
from modules.utils import detect_classes


def main() -> int:
    with open("config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)

    if not config.get("classes"):
        config["classes"] = detect_classes(config["train_dir"])

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        return 2

    image_path = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = len(config["classes"])
    model = build_model(config, num_classes)

    try:
        state_dict = torch.load(config["save_model_path"], map_location=device)
    except FileNotFoundError:
        print(f"Model not found: {config['save_model_path']}")
        return 1

    model.load_state_dict(state_dict)
    model.eval().to(device)

    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(tuple(config["img_size"])),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    top_k = int(config.get("top_k", 3))
    indices = probs.argsort()[::-1][:top_k]

    print("Top predictions:")
    for idx in indices:
        cls = config["classes"][idx]
        score = float(probs[idx])
        print(f"  {cls}: {score:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
