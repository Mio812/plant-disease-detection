"""Predict the disease class and estimate severity for a single image.

Usage:
    python -m scripts.predict --config configs/default.yaml \
        --checkpoint outputs/resnet18_best.pth --image path/to/leaf.jpg
"""

import argparse
import json

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import Config
from src.data import build_transforms, parse_class_name
from src.models import build_model
from src.audit.severity import estimate_severity
from src.utils import get_device, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Classify a leaf image and estimate severity.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--classes", default=None, help="Optional JSON file with the ordered class list."
    )
    return parser.parse_args()


def load_classes(path, fallback):
    if path is None:
        return fallback
    with open(path, encoding="utf-8") as f:
        return json.load(f)["classes"]


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    device = get_device()

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    classes = load_classes(args.classes, ckpt.get("classes"))
    if classes is None:
        raise ValueError("Class list not found; pass --classes <history.json>.")

    model = build_model(cfg.model.name, len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    transform = build_transforms(cfg.data.image_size, train=False)
    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    top_prob, top_idx = probs.max(0)
    class_name = classes[top_idx]
    crop, condition, is_healthy = parse_class_name(class_name)

    level, ratio = estimate_severity(
        cv2.imread(args.image), cfg.severity.thresholds, cfg.severity.levels
    )

    print(f"Crop:        {crop}")
    print(f"Condition:   {condition} ({'healthy' if is_healthy else 'diseased'})")
    print(f"Confidence:  {top_prob.item():.3f}")
    print(f"Lesion area: {ratio:.3f}")
    print(f"Severity:    {'healthy' if is_healthy else level}")


if __name__ == "__main__":
    main()
