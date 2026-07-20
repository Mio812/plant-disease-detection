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
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="checkpoint paths, parallel to --models; "
                             "defaults to outputs/<name>_best.pth")
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--plantdoc", nargs="+", default=["data/PlantDoc/test"])
    parser.add_argument("--image-size", type=int, default=None,
                        help="override the config; must match how the checkpoint was trained")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--save-reports", action="store_true",
                        help="also write the classification report and confusion matrix")
    return parser.parse_args()


def label_space(dataset, classes):
    """Classes the evaluation set can actually contain, or None for all of them."""
    if dataset != "plantdoc":
        return None
    from src.data.plantdoc import PLANTDOC_TO_PLANTVILLAGE
    allowed = set(PLANTDOC_TO_PLANTVILLAGE.values())
    return torch.tensor([c in allowed for c in classes])


def crop_ids(classes):
    """Map each class index to its crop, so species and disease can be scored apart."""
    crops = sorted({c.split("___")[0] for c in classes})
    lookup = {c: i for i, c in enumerate(crops)}
    return torch.tensor([lookup[c.split("___")[0]] for c in classes]), len(crops)


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
    if args.image_size:
        cfg.data.image_size = args.image_size
    set_seed(cfg.seed)
    device = get_device()
    out_dir = Path(cfg.output_dir)
    classes = ImageFolder(cfg.data.root).classes

    loader, n = build_eval_loader(cfg, args.dataset, args.plantdoc, classes)
    print(f"evaluating on '{args.dataset}' ({n} images)\n")

    probs, targets, results = [], None, {}
    paths = args.checkpoints or [out_dir / f"{n}_best.pth" for n in args.models]
    if len(paths) != len(args.models):
        raise SystemExit("--checkpoints must have one path per --models entry")
    for name, path in zip(args.models, paths):
        model = load_model(name, len(classes), path, device)
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

    # Where does the accuracy actually go? Score the species and the diagnosis apart,
    # and score again with impossible classes masked out.
    crop, n_crops = crop_ids(classes)
    crop_hit = crop[y_pred] == crop[y]
    results["ensemble_crop"] = round(float(crop_hit.float().mean() * 100), 2)
    results["ensemble_disease_given_crop"] = round(
        float((y_pred[crop_hit] == y[crop_hit]).float().mean() * 100), 2) if int(crop_hit.sum()) else 0.0
    results["n_crops"] = n_crops

    mask = label_space(args.dataset, classes)
    if mask is not None:
        restricted = ensemble.clone()
        restricted[:, ~mask] = float("-inf")
        y_restricted = restricted.argmax(1)
        results["ensemble_restricted"] = round(float((y_restricted == y).float().mean() * 100), 2)
        results["n_classes_possible"] = int(mask.sum())

    print(f"  {'binary h/d':14s} {results['ensemble_binary']:6.2f}%")
    print(f"  {'crop only':14s} {results['ensemble_crop']:6.2f}%   "
          f"(chance {100.0 / n_crops:.1f}%)")
    print(f"  {'disease|crop':14s} {results['ensemble_disease_given_crop']:6.2f}%")
    if mask is not None:
        print(f"  {'restricted':14s} {results['ensemble_restricted']:6.2f}%   "
              f"({int(mask.sum())} reachable classes)")
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
