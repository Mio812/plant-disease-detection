"""E26 (H17): train classification and severity together on one backbone.

E25 showed classification-only training discards the colour/texture detail severity
needs. This trains both objectives jointly so the backbone must keep it. Default
augmentation is standard, not strong -- strong ColorJitter randomises the colour the
severity target depends on. Bar: within-class rho beats Otsu's 0.78.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from src.audit.severity import lesion_ratio
from src.config import Config
from src.data import build_transforms, splits_from_config
from src.models.factory import build_model, strip_head
from src.training import build_strong_transforms
from src.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Joint classification + severity head (E26).")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default="resnet18")
    p.add_argument("--ratios", default="outputs/lesion_ratios.json")
    p.add_argument("--augment", choices=["standard", "strong"], default="standard")
    p.add_argument("--severity-weight", type=float, default=10.0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="outputs/joint_severity.json")
    return p.parse_args()


class JointModel(nn.Module):
    def __init__(self, name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = build_model(name, num_classes, pretrained=pretrained)
        dim = strip_head(self.backbone)
        self.class_head = nn.Linear(dim, num_classes)
        self.severity_head = nn.Linear(dim, 1)

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.class_head(f), self.severity_head(f).squeeze(1)


class JointDataset(Dataset):
    """Yields (image, class label, sqrt severity target). Target is -1 where the
    leaf has no cached ratio, so the severity loss can mask it out."""

    def __init__(self, items, ratios, transform):
        self.items = items
        self.ratios = ratios
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        r = self.ratios.get(Path(path).name)
        target = float(np.sqrt(r)) if r is not None else -1.0
        return self.transform(Image.open(path).convert("RGB")), label, target


def within_class(pred, official, labels):
    rhos, sizes = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) < 30:
            continue
        rho = spearmanr(pred[idx], official[idx]).statistic
        if np.isfinite(rho):
            rhos.append(rho)
            sizes.append(len(idx))
    if not rhos:
        return None, 0
    return round(float(np.average(rhos, weights=sizes)), 4), len(rhos)


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
    ratios = json.loads(Path(args.ratios).read_text(encoding="utf-8"))["ratios"]
    train_idx, _, test_idx = splits_from_config(cfg, base)
    if args.limit:
        train_idx, test_idx = train_idx[:args.limit], test_idx[:args.limit]

    train_tf = (build_strong_transforms(cfg.data.image_size) if args.augment == "strong"
                else build_transforms(cfg.data.image_size, train=True))
    eval_tf = build_transforms(cfg.data.image_size, train=False)
    train_items = [base.samples[i] for i in train_idx]
    test_items = [base.samples[i] for i in test_idx]

    train_dl = DataLoader(JointDataset(train_items, ratios, train_tf), batch_size=cfg.data.batch_size,
                          shuffle=True, num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True)
    test_dl = DataLoader(JointDataset(test_items, ratios, eval_tf), batch_size=cfg.data.batch_size,
                         num_workers=cfg.data.num_workers)

    model = JointModel(args.model, len(classes), pretrained=cfg.model.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"[E26] {args.model} joint, augment={args.augment}, severity_weight={args.severity_weight} "
          f"| train {len(train_items):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, sev in train_dl:
            x, y, sev = x.to(device), y.to(device), sev.to(device)
            logits, sev_pred = model(x)
            ce = F.cross_entropy(logits, y, label_smoothing=cfg.train.label_smoothing)
            has = sev >= 0
            mse = F.mse_loss(sev_pred[has].sigmoid(), sev[has]) if has.any() else logits.new_zeros(())
            loss = ce + args.severity_weight * mse
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        print(f"[{epoch:02d}/{args.epochs}] ce={float(ce):.3f} sev_mse={float(mse):.4f}", flush=True)

    model.eval()
    preds, sev_out, ys = [], [], []
    with torch.no_grad():
        for x, y, _ in test_dl:
            logits, sev_pred = model(x.to(device))
            preds.append(logits.argmax(1).cpu())
            sev_out.append(sev_pred.sigmoid().cpu())
            ys.append(y)
    preds, sev_out, ys = torch.cat(preds).numpy(), torch.cat(sev_out).numpy(), torch.cat(ys).numpy()
    class_acc = round(float((preds == ys).mean() * 100), 2)

    official = np.array([ratios.get(Path(p).name, 0.0) for p, _ in test_items])
    diseased = np.array([0 if "healthy" in classes[l].lower() else 1 for _, l in test_items])
    otsu = np.array([lesion_ratio(cv2.imread(p)) for p, _ in test_items], dtype=np.float32)

    rho_head, n_scored = within_class(sev_out, official, ys)
    rho_otsu, _ = within_class(otsu, official, ys)
    summary = {
        "augment": args.augment, "severity_weight": args.severity_weight,
        "classification_accuracy": class_acc,
        "within_class_rho_head": rho_head, "within_class_rho_otsu": rho_otsu,
        "within_class_n_classes": n_scored,
        "auc_head": round(float(roc_auc_score(diseased, sev_out)), 4),
        "auc_otsu": round(float(roc_auc_score(diseased, otsu)), 4),
    }
    fmt = lambda v: f"{v:.3f}" if v is not None else "n/a"
    print(f"\n[E26] classification {class_acc:.2f}%   "
          f"within-class rho: head {fmt(rho_head)} vs Otsu {fmt(rho_otsu)}   (bar: beat Otsu)")
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    torch.save({"model": model.state_dict(), "classes": classes}, Path(cfg.output_dir) / "joint_model.pth")
    preds_out = [{"name": Path(p).name, "class": classes[l], "head": float(h),
                  "official": float(o), "otsu": float(t)}
                 for (p, l), h, o, t in zip(test_items, sev_out, official, otsu)]
    (Path(cfg.output_dir) / "joint_predictions.json").write_text(json.dumps(preds_out), encoding="utf-8")
    print(f"saved {args.out}, joint_model.pth, joint_predictions.json")


if __name__ == "__main__":
    main()
