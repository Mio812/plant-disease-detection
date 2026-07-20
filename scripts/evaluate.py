"""Score checkpoints on any evaluation set.

    --on plantvillage   held-out PlantVillage test split
    --on segmented      the same leaves with the background removed
    --on grayscale      the same leaves without colour
    --on plantdoc       in-the-wild field photographs

Usage:
    python -m scripts.evaluate --on plantvillage --weights 0.2 0.5 0.3
    python -m scripts.evaluate --on plantdoc --plantdoc data/PlantDoc/test data/PlantDoc/train
"""

import argparse
import json
from pathlib import Path

import torch

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.config import Config
from src.data import (
    ItemDataset,
    build_transforms,
    plantdoc_items,
    splits_from_config,
    variant_root,
    variant_samples,
    wilson_interval,
)
from src.evaluation import (
    compute_metrics,
    confusion,
    full_report,
    load_model,
    predict_loader,
)
from src.models import combine
from src.utils import get_device, set_seed

MEMBERS = ["custom_cnn", "resnet18", "mobilenet_v2"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on any dataset.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--on", dest="dataset",
                        choices=["plantvillage", "segmented", "grayscale", "plantdoc"],
                        default="plantvillage")
    parser.add_argument("--models", nargs="+", default=MEMBERS)
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--plantdoc", nargs="+", default=["data/PlantDoc/test"])
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--save-reports", action="store_true",
                        help="also write the classification report and confusion matrix")
    return parser.parse_args()


def build_eval_loader(cfg, dataset, plantdoc_dirs, classes):
    """One loader builder for every evaluation set."""
    transform = build_transforms(cfg.data.image_size, train=False)
    base = ImageFolder(cfg.data.root)
    if dataset == "plantdoc":
        items = []
        for directory in plantdoc_dirs:
            found = plantdoc_items(directory, {c: i for i, c in enumerate(classes)})
            print(f"  {directory}: {len(found)} mapped images")
            items.extend(found)
    else:
        variant = "color" if dataset == "plantvillage" else dataset
        _, _, test_idx = splits_from_config(cfg, len(base))
        items = variant_samples(base, test_idx, variant, variant_root(cfg.data.root, variant))
    loader = DataLoader(ItemDataset(items, transform), batch_size=cfg.data.batch_size,
                        num_workers=cfg.data.num_workers)
    return loader, len(items)


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    device = get_device()
    out_dir = Path(cfg.output_dir)
    classes = ImageFolder(cfg.data.root).classes

    loader, n = build_eval_loader(cfg, args.dataset, args.plantdoc, classes)
    print(f"evaluating on '{args.dataset}' ({n} images)\n")

    probs, targets, results = [], None, {}
    for name in args.models:
        model = load_model(name, len(classes), out_dir / f"{name}_best.pth", device)
        member, targets = predict_loader(model, loader, device, tta=args.tta)
        probs.append(member)
        correct = int((member.argmax(1) == torch.tensor(targets)).sum())
        low, high = wilson_interval(correct, n)
        results[name] = round(100.0 * correct / n, 2)
        print(f"  {name:14s} {results[name]:6.2f}%   95% CI [{low * 100:.1f}, {high * 100:.1f}]")

    y = torch.tensor(targets)
    ensemble = combine(probs, args.weights)
    y_pred = ensemble.argmax(1)
    correct = int((y_pred == y).sum())
    low, high = wilson_interval(correct, n)
    results["ensemble"] = round(100.0 * correct / n, 2)
    print(f"  {'ensemble':14s} {results['ensemble']:6.2f}%   95% CI [{low * 100:.1f}, {high * 100:.1f}]")

    healthy = torch.tensor([1 if "healthy" in c.lower() else 0 for c in classes])
    results["ensemble_binary"] = round(
        float((healthy[y_pred] == healthy[y]).float().mean() * 100), 2)
    print(f"  {'binary h/d':14s} {results['ensemble_binary']:6.2f}%")
    results["n_images"] = n
    results["dataset"] = args.dataset

    if args.save_reports:
        y_true, y_hat = y.tolist(), y_pred.tolist()
        results["metrics"] = compute_metrics(y_true, y_hat)
        (out_dir / f"ensemble_{args.dataset}_report.txt").write_text(
            full_report(y_true, y_hat, classes), encoding="utf-8")
        cm = confusion(y_true, y_hat, len(classes))
        pairs = [{"true": classes[i], "pred": classes[j], "count": int(cm[i, j])}
                 for i in range(len(classes)) for j in range(len(classes))
                 if i != j and cm[i, j] > 0]
        pairs.sort(key=lambda p: p["count"], reverse=True)
        (out_dir / f"ensemble_{args.dataset}_confused_pairs.json").write_text(
            json.dumps({"total_errors": sum(p["count"] for p in pairs), "pairs": pairs}, indent=2),
            encoding="utf-8")

    out = Path(args.out) if args.out else out_dir / f"eval_{args.dataset}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved {out.as_posix()}")


if __name__ == "__main__":
    main()
