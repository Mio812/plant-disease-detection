"""Train against the PlantVillage background shortcut and measure field transfer.

Two interventions can be compared under an identical split:

* ``--variant color --p-random P`` composites each leaf onto a random background
  with probability P (the leaf keeps a realistic surround).
* ``--variant segmented`` trains directly on the background-removed images
  (leaf on black), which is also the clean control for the segmented evaluation.

Both report PlantVillage and PlantDoc accuracy.

Usage:
    python -m scripts.train_robust --model resnet18 --image-size 224 --p-random 0.7
    python -m scripts.train_robust --model resnet18 --image-size 224 --variant segmented
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.augment import BackgroundRandomised, build_strong_transforms
from src.config import Config
from src.crossdata import plantdoc_items
from src.data import ItemDataset, build_transforms, variant_root, variant_samples
from src.engine import evaluate, fit
from src.metrics import full_report
from src.models import build_model
from src.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Robust training against the background shortcut.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--variant", choices=["color", "grayscale", "segmented"], default="color")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--p-random", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--plantdoc", default="data/PlantDoc/test")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    if args.image_size:
        cfg.data.image_size = args.image_size
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.variant != "color":
        args.p_random = 0.0
    set_seed(cfg.seed)
    device = get_device()

    base = ImageFolder(cfg.data.root)
    segmented_root = variant_root(cfg.data.root, "segmented")
    active_root = variant_root(cfg.data.root, args.variant)
    n = len(base)
    n_test = int(n * cfg.data.test_split)
    n_val = int(n * cfg.data.val_split)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

    eval_tf = build_transforms(cfg.data.image_size, train=False)
    val_items = variant_samples(base, val_idx, args.variant, active_root)
    test_items = variant_samples(base, test_idx, args.variant, active_root)

    if args.variant != "color":
        train_ds = ItemDataset(variant_samples(base, train_idx, args.variant, active_root),
                               build_strong_transforms(cfg.data.image_size))
    else:
        train_ds = BackgroundRandomised([base.samples[i] for i in train_idx], base.classes,
                                        segmented_root, cfg.data.image_size,
                                        p_random=args.p_random, seed=cfg.seed)

    loaders = {
        "train": DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True,
                            num_workers=cfg.data.num_workers, pin_memory=True),
        "val": DataLoader(ItemDataset(val_items, eval_tf), batch_size=cfg.data.batch_size,
                          num_workers=cfg.data.num_workers),
        "test": DataLoader(ItemDataset(test_items, eval_tf), batch_size=cfg.data.batch_size,
                           num_workers=cfg.data.num_workers),
    }
    print(f"variant={args.variant} p_random={args.p_random} size={cfg.data.image_size} | "
          f"train={len(train_ds)} val={len(val_items)} test={len(test_items)}")

    classes = base.classes
    model = build_model(args.model, len(classes), cfg.model.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{args.variant}_p{int(args.p_random * 100)}_{cfg.data.image_size}"
    history = fit(model, loaders, criterion, optimizer, scheduler, device,
                  cfg.train.epochs, cfg.train.early_stop_patience, out_dir / f"{tag}_best.pth")

    test_metrics, y_true, y_pred = evaluate(model, loaders["test"], criterion, device)
    print(f"\nPlantVillage ({args.variant}) test:", {k: round(v, 4) for k, v in test_metrics.items()})
    print(full_report(y_true, y_pred, classes))

    items = plantdoc_items(args.plantdoc, {c: i for i, c in enumerate(classes)})
    images = torch.stack([eval_tf(Image.open(p).convert("RGB")) for p, _ in items])
    targets = torch.tensor([y for _, y in items])
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(images[i:i + 64].to(device)).argmax(1).cpu()
                           for i in range(0, len(images), 64)])
    field_acc = float((preds == targets).float().mean() * 100)
    print(f"PlantDoc (field) accuracy: {field_acc:.2f}%  [{tag}]")

    with open(out_dir / f"{tag}_history.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "test_metrics": test_metrics,
                   "plantdoc_accuracy": field_acc, "variant": args.variant,
                   "p_random": args.p_random, "image_size": cfg.data.image_size,
                   "classes": classes}, f, indent=2)


if __name__ == "__main__":
    main()
