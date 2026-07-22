"""E21 (H11): does field accuracy hold across capture conditions?

Bins the field images by luminance, contrast and sharpness, and scores each arm
per bin. If the strong-augmentation arm degrades less across the tails than the
standard-augmentation baseline, H11 holds -- with the caveat that the only
standard-aug arm is 128px, so that comparison also varies resolution.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from src.config import Config
from src.data import build_transforms, plantdoc_items
from src.evaluation import load_model
from src.utils import get_device, set_seed

ARMS = [
    ("resnet18_color_strong_p70_224_best.pth", 224, "strong aug (bg-random)"),
    ("resnet18_best.pth", 128, "standard aug (baseline)"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Field robustness by capture quality (E21).")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--plantdoc", nargs="+", default=["data/PlantDoc/test", "data/PlantDoc/train"])
    p.add_argument("--bins", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="outputs/field_robustness.json")
    return p.parse_args()


def capture_metrics(path):
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {"luminance": float(gray.mean()),
            "contrast": float(gray.std()),
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var())}


@torch.no_grad()
def predict(checkpoint, size, items, classes, device):
    model = load_model("resnet18", len(classes), f"outputs/{checkpoint}", device)
    tf = build_transforms(size, train=False)
    preds = []
    for i in range(0, len(items), 64):
        batch = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in items[i:i + 64]])
        preds.append(model(batch.to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy()


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    device = get_device()
    classes = ImageFolder(cfg.data.root).classes

    items = []
    for d in args.plantdoc:
        items.extend(plantdoc_items(d, {c: i for i, c in enumerate(classes)}))
    if args.limit:
        items = items[:args.limit]

    metrics = [capture_metrics(p) for p, _ in items]
    keep = [i for i, m in enumerate(metrics) if m is not None]
    items = [items[i] for i in keep]
    metrics = [metrics[i] for i in keep]
    y = np.array([lab for _, lab in items])
    print(f"{len(items):,} field images with capture metrics")

    arm_preds = {label: predict(ckpt, size, items, classes, device)
                 for ckpt, size, label in ARMS}

    edges = np.linspace(0, 100, args.bins + 1)[1:-1]
    summary = {}
    for metric in ("luminance", "contrast", "sharpness"):
        vals = np.array([m[metric] for m in metrics])
        cuts = np.percentile(vals, edges)
        bin_id = np.digitize(vals, cuts)
        summary[metric] = {}
        print(f"\n== {metric} ==")
        for label, preds in arm_preds.items():
            accs = [round(100.0 * float((preds[bin_id == b] == y[bin_id == b]).mean()), 2)
                    for b in range(args.bins)]
            gap = round(accs[-1] - accs[0], 2)          # high-quality bin minus low-quality bin
            summary[metric][label] = {"per_bin_accuracy": accs, "high_minus_low": gap}
            print(f"  {label:26s} bins(low..high) {accs}   gap {gap:+.2f}")

    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
