"""Train with background randomisation and strong augmentation.

Aims to close the laboratory-to-field gap: each training leaf is composited onto
a random background so the network cannot rely on PlantVillage's capture bias.
Reports accuracy on both the PlantVillage test split and PlantDoc field images.

Usage:
    python -m scripts.train_robust --model resnet18 --image-size 224 --epochs 15
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from src.augment import BackgroundRandomised
from src.config import Config
from src.crossdata import plantdoc_items
from src.data import TransformSubset, build_transforms
from src.engine import evaluate, fit
from src.metrics import full_report
from src.models import build_model
from src.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Robust training against background shortcut.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="resnet18")
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
    set_seed(cfg.seed)
    device = get_device()

    base = ImageFolder(cfg.data.root)
    n = len(base)
    n_test = int(n * cfg.data.test_split)
    n_val = int(n * cfg.data.val_split)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

    eval_tf = build_transforms(cfg.data.image_size, train=False)
    train_ds = BackgroundRandomised([base.samples[i] for i in train_idx], base.classes,
                                    cfg.data.root.replace("/color", "/segmented"),
                                    cfg.data.image_size, p_random=args.p_random, seed=cfg.seed)
    loaders = {
        "train": DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True,
                            num_workers=cfg.data.num_workers, pin_memory=True),
        "val": DataLoader(TransformSubset(Subset(base, val_idx), eval_tf),
                          batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers),
        "test": DataLoader(TransformSubset(Subset(base, test_idx), eval_tf),
                           batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers),
    }

    classes = base.classes
    model = build_model(args.model, len(classes), cfg.model.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_robust_p{int(args.p_random * 100)}_{cfg.data.image_size}"
    history = fit(model, loaders, criterion, optimizer, scheduler, device,
                  cfg.train.epochs, cfg.train.early_stop_patience, out_dir / f"{tag}_best.pth")

    test_metrics, y_true, y_pred = evaluate(model, loaders["test"], criterion, device)
    print("\nPlantVillage test:", {k: round(v, 4) for k, v in test_metrics.items()})
    print(full_report(y_true, y_pred, classes))

    items = plantdoc_items(args.plantdoc, {c: i for i, c in enumerate(classes)})
    images = torch.stack([eval_tf(Image.open(p).convert("RGB")) for p, _ in items])
    targets = torch.tensor([y for _, y in items])
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(images[i:i + 64].to(device)).argmax(1).cpu()
                           for i in range(0, len(images), 64)])
    field_acc = float((preds == targets).float().mean() * 100)
    print(f"PlantDoc (field) accuracy: {field_acc:.2f}%")

    with open(out_dir / f"{tag}_history.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "test_metrics": test_metrics,
                   "plantdoc_accuracy": field_acc, "p_random": args.p_random,
                   "image_size": cfg.data.image_size, "classes": classes}, f, indent=2)


if __name__ == "__main__":
    main()
