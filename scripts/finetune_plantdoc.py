"""Few-shot supervised adaptation to field images.

Starts from a PlantVillage-trained checkpoint and fine-tunes on the PlantDoc
train split, then reports accuracy on the PlantDoc test split. This is
*supervised* domain adaptation and must be reported separately from the
zero-shot numbers.

Usage:
    python -m scripts.finetune_plantdoc --model resnet18 \
        --checkpoint outputs/resnet18_best.pth --shots 20 --epochs 15
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from src.augment import build_strong_transforms
from src.config import Config
from src.crossdata import plantdoc_items
from src.data import build_transforms
from src.models import build_model
from src.utils import get_device, load_checkpoint, set_seed


class ItemDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune on PlantDoc field images.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--checkpoint", default="outputs/resnet18_best.pth")
    parser.add_argument("--train-dir", default="data/PlantDoc/train")
    parser.add_argument("--test-dir", default="data/PlantDoc/test")
    parser.add_argument("--shots", type=int, default=0, help="images per class; 0 = use all")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    if args.image_size:
        cfg.data.image_size = args.image_size
    set_seed(cfg.seed)
    device = get_device()

    classes = ImageFolder(cfg.data.root).classes
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_items = plantdoc_items(args.train_dir, class_to_idx)
    test_items = plantdoc_items(args.test_dir, class_to_idx)

    if args.shots:
        by_class = defaultdict(list)
        for item in train_items:
            by_class[item[1]].append(item)
        rng = random.Random(cfg.seed)
        train_items = [it for group in by_class.values()
                       for it in rng.sample(group, min(args.shots, len(group)))]
    print(f"fine-tune on {len(train_items)} field images | test on {len(test_items)}")

    train_tf = build_strong_transforms(cfg.data.image_size)
    eval_tf = build_transforms(cfg.data.image_size, train=False)
    train_dl = DataLoader(ItemDataset(train_items, train_tf), batch_size=32, shuffle=True,
                          num_workers=cfg.data.num_workers, drop_last=False)
    test_dl = DataLoader(ItemDataset(test_items, eval_tf), batch_size=64,
                         num_workers=cfg.data.num_workers)

    model = build_model(args.model, len(classes), pretrained=False).to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint, map_location=device)["model"])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def field_accuracy():
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_dl:
                correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
                total += y.numel()
        return 100.0 * correct / total

    before = field_accuracy()
    print(f"PlantDoc accuracy before fine-tuning: {before:.2f}%")

    best, history = before, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = field_accuracy()
        history.append({"epoch": epoch, "plantdoc_accuracy": acc})
        print(f"[{epoch:02d}/{args.epochs}] PlantDoc accuracy {acc:.2f}%")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "plantdoc_accuracy": acc},
                       Path(cfg.output_dir) / f"{args.model}_plantdoc_ft.pth")

    print(f"\nbest PlantDoc accuracy {best:.2f}%  (zero-shot start {before:.2f}%)")
    out = Path(cfg.output_dir) / f"{args.model}_plantdoc_ft_history.json"
    out.write_text(json.dumps({"shots": args.shots, "before": before, "best": best,
                               "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
