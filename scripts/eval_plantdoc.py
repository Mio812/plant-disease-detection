"""Zero-shot evaluation on PlantDoc field images.

Models trained on PlantVillage are scored, without any fine-tuning, on
in-the-wild photographs. The gap against the PlantVillage test accuracy is the
headline generalisation result.

Usage:
    python -m scripts.eval_plantdoc --config configs/default.yaml
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from src.config import Config
from src.crossdata import plantdoc_items
from src.data import build_transforms
from src.ensemble import combine
from src.models import build_model
from src.utils import get_device, load_checkpoint, set_seed

MEMBERS = ["custom_cnn", "resnet18", "mobilenet_v2"]


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot PlantDoc evaluation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--plantdoc", default="data/PlantDoc/test")
    parser.add_argument("--weights", type=float, nargs=3, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    device = get_device()
    out_dir = Path(cfg.output_dir)

    classes = ImageFolder(cfg.data.root).classes
    items = plantdoc_items(args.plantdoc, {c: i for i, c in enumerate(classes)})
    transform = build_transforms(cfg.data.image_size, train=False)
    images = torch.stack([transform(Image.open(p).convert("RGB")) for p, _ in items])
    targets = torch.tensor([y for _, y in items])
    print(f"PlantDoc: {len(items)} field images over {len(set(targets.tolist()))} mapped classes")

    probs, results = [], {}
    for name in MEMBERS:
        model = build_model(name, len(classes), pretrained=False).to(device)
        model.load_state_dict(load_checkpoint(out_dir / f"{name}_best.pth", map_location=device)["model"])
        model.eval()
        with torch.no_grad():
            member = torch.cat([torch.softmax(model(images[i:i + 64].to(device)), 1).cpu()
                                for i in range(0, len(images), 64)])
        probs.append(member)
        results[name] = float((member.argmax(1) == targets).float().mean() * 100)
        print(f"  {name:14s} PlantDoc accuracy {results[name]:.2f}%")

    ensemble = combine(probs, args.weights)
    results["ensemble"] = float((ensemble.argmax(1) == targets).float().mean() * 100)
    healthy = torch.tensor([1 if "healthy" in c.lower() else 0 for c in classes])
    results["ensemble_binary"] = float(
        (healthy[ensemble.argmax(1)] == healthy[targets]).float().mean() * 100
    )
    print(f"  {'ensemble':14s} PlantDoc accuracy {results['ensemble']:.2f}%")
    print(f"  healthy/diseased binary accuracy {results['ensemble_binary']:.2f}%")
    (out_dir / "plantdoc_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
