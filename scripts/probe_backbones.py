"""E24: how far does field accuracy track the frozen backbone alone?

E15 showed that training 19,494 parameters and training 11.2M reach the same
field accuracy, so whatever transfers is already in the pretrained features.
This walks a ladder of frozen CNNs, moving one factor per rung -- capacity, then
training recipe, then architecture generation, then pretraining data.

Features are extracted once per backbone and the head is fitted on the cache,
which is what makes five rungs affordable. That means no augmentation reaches
the backbone, so this is a different protocol from the E15 frozen arms and
ResNet-18 is re-run here to serve as the control under matched conditions.

Usage:
    python -m scripts.probe_backbones
    python -m scripts.probe_backbones --backbones resnet18:IMAGENET1K_V1
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder

from src.config import Config
from src.data import (
    ItemDataset,
    build_transforms,
    plantdoc_items,
    splits_from_config,
    wilson_interval,
)
from src.models.hierarchical import crop_of
from src.utils import get_device, set_seed

LADDER = [
    "resnet18:IMAGENET1K_V1",
    "resnet50:IMAGENET1K_V1",
    "resnet50:IMAGENET1K_V2",
    "convnext_tiny:IMAGENET1K_V1",
    "regnet_y_16gf:IMAGENET1K_SWAG_LINEAR_V1",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen-backbone ladder (E24).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbones", nargs="+", default=LADDER,
                        help="name:weight_tag entries")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--plantdoc", nargs="+",
                        default=["data/PlantDoc/test", "data/PlantDoc/train"])
    parser.add_argument("--out", default="outputs/probe_backbones.json")
    return parser.parse_args()


def strip_head(model):
    """Replace the final Linear with Identity and return its input width.

    Families disagree on head layout: resnet and regnet expose `fc`, convnext and
    efficientnet wrap the Linear inside `classifier`, densenet uses a bare Linear.
    """
    for attr in ("fc", "classifier", "head"):
        head = getattr(model, attr, None)
        if isinstance(head, nn.Linear):
            setattr(model, attr, nn.Identity())
            return head.in_features
        if isinstance(head, nn.Sequential):
            for i in range(len(head) - 1, -1, -1):
                if isinstance(head[i], nn.Linear):
                    width = head[i].in_features
                    head[i] = nn.Identity()
                    return width
    raise SystemExit(f"no final Linear in {type(model).__name__}")


@torch.no_grad()
def extract(backbone, loader, device):
    feats, labels = [], []
    for images, targets in loader:
        feats.append(backbone(images.to(device)).flatten(1).cpu())
        labels.extend(targets.tolist())
    return torch.cat(feats), torch.tensor(labels)


def fit_head(train, val, num_classes, epochs, lr, device):
    """Train the linear head on cached features, keeping the best epoch on val."""
    x, y = train[0].to(device), train[1].to(device)
    xv, yv = val[0].to(device), val[1].to(device)
    head = nn.Linear(x.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, best_state = 0.0, None
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(x), device=device)
        for i in range(0, len(x), 1024):
            idx = perm[i:i + 1024]
            opt.zero_grad()
            crit(head(x[idx]), y[idx]).backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            acc = float((head(xv).argmax(1) == yv).float().mean())
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, best_acc


@torch.no_grad()
def score(head, cache, classes, crop_index, device):
    x, y = cache[0].to(device), cache[1].to(device)
    pred = head(x).argmax(1)
    hit = int((pred == y).sum())
    n = len(y)
    healthy = torch.tensor([1 if "healthy" in c.lower() else 0 for c in classes], device=device)
    crop = crop_index.to(device)
    crop_hit = crop[pred] == crop[y]
    low, high = wilson_interval(hit, n)
    return {
        "n": n,
        "accuracy": round(100.0 * hit / n, 2),
        "ci_low": round(low * 100, 2),
        "ci_high": round(high * 100, 2),
        "binary": round(float((healthy[pred] == healthy[y]).float().mean() * 100), 2),
        "crop": round(float(crop_hit.float().mean() * 100), 2),
        "disease_given_crop": round(
            float((pred[crop_hit] == y[crop_hit]).float().mean() * 100), 2) if int(crop_hit.sum()) else 0.0,
    }


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    cfg.data.image_size = args.image_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    set_seed(cfg.seed)
    device = get_device()

    base = ImageFolder(cfg.data.root)
    classes = base.classes
    crop_index, crops = crop_of(classes)
    train_idx, val_idx, test_idx = splits_from_config(cfg, len(base))
    transform = build_transforms(cfg.data.image_size, train=False)

    field_items = []
    for directory in args.plantdoc:
        field_items.extend(plantdoc_items(directory, {c: i for i, c in enumerate(classes)}))

    def loader(items):
        return DataLoader(ItemDataset(items, transform), batch_size=cfg.data.batch_size,
                          num_workers=cfg.data.num_workers, pin_memory=True)

    loaders = {
        "train": loader([base.samples[i] for i in train_idx]),
        "val": loader([base.samples[i] for i in val_idx]),
        "test": loader([base.samples[i] for i in test_idx]),
        "field": loader(field_items),
    }
    print(f"{len(classes)} classes / {len(crops)} crops | field images {len(field_items)}")

    results = {}
    for spec in args.backbones:
        name, _, tag = spec.partition(":")
        backbone = models.get_model(name, weights=tag or None)
        width = strip_head(backbone)
        backbone = backbone.to(device).eval()
        frozen = sum(p.numel() for p in backbone.parameters())
        print(f"\n[{spec}] frozen {frozen:,} parameters, {width}-d features")

        cache = {split: extract(backbone, dl, device) for split, dl in loaders.items()}
        del backbone
        torch.cuda.empty_cache()

        head, val_acc = fit_head(cache["train"], cache["val"], len(classes),
                                 args.epochs, args.lr, device)
        lab = score(head, cache["test"], classes, crop_index, device)
        field = score(head, cache["field"], classes, crop_index, device)
        results[spec] = {
            "feature_dim": width,
            "frozen_parameters": frozen,
            "head_parameters": width * len(classes) + len(classes),
            "val_accuracy": round(val_acc * 100, 2),
            "plantvillage": lab,
            "plantdoc": field,
        }
        print(f"[{spec}] lab {lab['accuracy']:.2f}%   field {field['accuracy']:.2f}% "
              f"[{field['ci_low']:.1f}, {field['ci_high']:.1f}]   "
              f"binary {field['binary']:.2f}%   crop {field['crop']:.2f}%")

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
