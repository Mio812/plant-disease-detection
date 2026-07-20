"""Train one classifier.

    --variant color --augment standard              baseline
    --variant color --augment strong --p-random 0.7 background randomisation
    --variant segmented                             trained without background
    --variant grayscale                             colour-cue ablation

Reports PlantVillage and, when available, PlantDoc accuracy.

Usage:
    python -m scripts.train --model resnet18
    python -m scripts.train --model resnet18 --augment strong --p-random 0.7 --image-size 224
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
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
)
from src.evaluation import full_report
from src.models import (
    HierarchicalClassifier, LogProbLoss, build_model, crop_of,
    freeze_backbone, trainable_parameters,
)
from src.training import BackgroundRandomised, build_strong_transforms, evaluate, fit
from src.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a plant-disease classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--variant", choices=["color", "grayscale", "segmented"], default="color")
    parser.add_argument("--augment", choices=["standard", "strong"], default="standard")
    parser.add_argument("--p-random", type=float, default=0.0,
                        help="probability of replacing the background (colour variant only)")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="lower this when running several arms concurrently; "
                             "Windows spawns a full process per worker")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="train only the classification head (linear probe)")
    parser.add_argument("--head", choices=["flat", "hierarchical"], default="flat",
                        help="hierarchical factorises p(class) into p(crop) p(class|crop)")
    parser.add_argument("--plantdoc", default="data/PlantDoc/test")
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def resolve_tag(args, cfg):
    if args.tag:
        return args.tag
    if (args.variant == "color" and args.augment == "standard"
            and args.p_random == 0.0 and not args.freeze_backbone
            and args.head == "flat"):
        return cfg.model.name
    suffix = ("_hier" if args.head == "hierarchical" else "") + \
             ("_frozen" if args.freeze_backbone else "")
    return (f"{cfg.model.name}_{args.variant}_{args.augment}"
            f"_p{int(args.p_random * 100)}_{cfg.data.image_size}{suffix}")


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    if args.model:
        cfg.model.name = args.model
    if args.image_size:
        cfg.data.image_size = args.image_size
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.lr:
        cfg.train.lr = args.lr
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.variant != "color":
        args.p_random = 0.0
    set_seed(cfg.seed)
    device = get_device()

    base = ImageFolder(cfg.data.root)
    train_idx, val_idx, test_idx = splits_from_config(cfg, len(base))
    active_root = variant_root(cfg.data.root, args.variant)
    eval_tf = build_transforms(cfg.data.image_size, train=False)
    train_tf = (build_strong_transforms(cfg.data.image_size) if args.augment == "strong"
                else build_transforms(cfg.data.image_size, train=True))

    if args.p_random > 0:
        train_ds = BackgroundRandomised([base.samples[i] for i in train_idx], base.classes,
                                        variant_root(cfg.data.root, "segmented"),
                                        cfg.data.image_size, p_random=args.p_random, seed=cfg.seed)
    else:
        train_ds = ItemDataset(variant_samples(base, train_idx, args.variant, active_root), train_tf)

    loaders = {
        "train": DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True,
                            num_workers=cfg.data.num_workers, pin_memory=True),
        "val": DataLoader(ItemDataset(variant_samples(base, val_idx, args.variant, active_root), eval_tf),
                          batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers),
        "test": DataLoader(ItemDataset(variant_samples(base, test_idx, args.variant, active_root), eval_tf),
                           batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers),
    }

    tag = resolve_tag(args, cfg)
    classes = base.classes
    print(f"[{tag}] variant={args.variant} augment={args.augment} p_random={args.p_random} "
          f"size={cfg.data.image_size} | train={len(train_ds)}")

    if args.head == "hierarchical":
        crop_index, crops = crop_of(classes)
        model = HierarchicalClassifier(cfg.model.name, crop_index, cfg.model.pretrained)
        print(f"[{tag}] hierarchical head: {len(crops)} crops x {len(classes)} classes")
    else:
        model = build_model(cfg.model.name, len(classes), cfg.model.pretrained)
    model = model.to(device)
    if args.freeze_backbone:
        if not cfg.model.pretrained:
            raise SystemExit("--freeze-backbone needs pretrained weights to probe")
        freeze_backbone(model, cfg.model.name)
    trainable, total = trainable_parameters(model)
    print(f"[{tag}] trainable {trainable:,} / {total:,} parameters "
          f"({100.0 * trainable / total:.2f}%)")

    criterion = (LogProbLoss(cfg.train.label_smoothing) if args.head == "hierarchical"
                 else nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = fit(model, loaders, criterion, optimizer, scheduler, device,
                  cfg.train.epochs, cfg.train.early_stop_patience, out_dir / f"{tag}_best.pth")

    test_metrics, y_true, y_pred = evaluate(model, loaders["test"], criterion, device)
    print(f"\nPlantVillage ({args.variant}) test:", {k: round(v, 4) for k, v in test_metrics.items()})
    print(full_report(y_true, y_pred, classes))

    field_acc = None
    if Path(args.plantdoc).exists():
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
                   "plantdoc_accuracy": field_acc, "variant": args.variant,
                   "augment": args.augment, "p_random": args.p_random,
                   "frozen_backbone": args.freeze_backbone, "head": args.head,
                   "lr": cfg.train.lr,
                   "trainable_parameters": trainable, "total_parameters": total,
                   "image_size": cfg.data.image_size, "classes": classes}, f, indent=2)


if __name__ == "__main__":
    main()
