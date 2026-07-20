"""Zero-shot evaluation on PlantDoc field images.

Models trained on PlantVillage are scored, without fine-tuning, on in-the-wild
photographs. Because the model has never seen any PlantDoc image, both official
splits are valid unseen data: pass several directories to evaluate on all of
PlantDoc for a much tighter confidence interval.

Usage:
    python -m scripts.eval_plantdoc --weights 0.2 0.5 0.3
    python -m scripts.eval_plantdoc --plantdoc data/PlantDoc/test data/PlantDoc/train
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from src.config import Config
from src.crossdata import plantdoc_items, wilson_interval
from src.data import build_transforms
from src.ensemble import combine
from src.models import build_model
from src.utils import get_device, load_checkpoint, set_seed

MEMBERS = ["custom_cnn", "resnet18", "mobilenet_v2"]


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot PlantDoc evaluation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--plantdoc", nargs="+", default=["data/PlantDoc/test"])
    parser.add_argument("--weights", type=float, nargs=3, default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def report(name, correct, total):
    accuracy = 100.0 * correct / total
    low, high = wilson_interval(correct, total)
    print(f"  {name:14s} {accuracy:5.2f}%   95% CI [{low * 100:.1f}, {high * 100:.1f}]")
    return accuracy


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    device = get_device()
    out_dir = Path(cfg.output_dir)

    classes = ImageFolder(cfg.data.root).classes
    class_to_idx = {c: i for i, c in enumerate(classes)}
    items = []
    for directory in args.plantdoc:
        found = plantdoc_items(directory, class_to_idx)
        print(f"{directory}: {len(found)} mapped images")
        items.extend(found)

    transform = build_transforms(cfg.data.image_size, train=False)
    images = torch.stack([transform(Image.open(p).convert("RGB")) for p, _ in items])
    targets = torch.tensor([y for _, y in items])
    print(f"total {len(items)} field images over {len(set(targets.tolist()))} mapped classes\n")

    probs, results = [], {}
    for name in MEMBERS:
        model = build_model(name, len(classes), pretrained=False).to(device)
        model.load_state_dict(load_checkpoint(out_dir / f"{name}_best.pth", map_location=device)["model"])
        model.eval()
        with torch.no_grad():
            member = torch.cat([torch.softmax(model(images[i:i + 64].to(device)), 1).cpu()
                                for i in range(0, len(images), 64)])
        probs.append(member)
        results[name] = report(name, int((member.argmax(1) == targets).sum()), len(targets))

    ensemble = combine(probs, args.weights)
    results["ensemble"] = report("ensemble", int((ensemble.argmax(1) == targets).sum()), len(targets))

    healthy = torch.tensor([1 if "healthy" in c.lower() else 0 for c in classes])
    binary_correct = int((healthy[ensemble.argmax(1)] == healthy[targets]).sum())
    results["ensemble_binary"] = report("binary h/d", binary_correct, len(targets))
    results["n_images"] = len(items)

    out = Path(args.out) if args.out else out_dir / "plantdoc_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved {out.as_posix()}")


if __name__ == "__main__":
    main()
