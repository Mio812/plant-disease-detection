"""E25: severity as an output of the classifier rather than a pipeline beside it.

Brief task 2 asks to *expand the model* to estimate severity from image features.
The existing estimator does not touch the network -- it thresholds HSV colour --
so this attaches a severity head to a trained classification backbone and
regresses the lesion ratio measured from the official segmented masks.

The point is what happens at inference. Otsu has to re-derive the leaf mask from
the image and loses accuracy doing it; a head that learned the mask implicitly
needs no mask at all. Both bounds are measured on the same test split here, so
the arm is falsifiable on arrival: it has to beat Otsu to be worth having.

The target is sqrt(ratio). Raw lesion ratio is heavily skewed, and sqrt is
monotone, so ranking metrics are untouched while the optimiser gets a far better
conditioned target.

Usage:
    python -m scripts.train_severity
    python -m scripts.train_severity --limit 400        # smoke test
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.audit.severity import lesion_ratio
from src.config import Config
from src.data import ItemDataset, build_transforms, splits_from_config
from src.models.factory import build_model, strip_head
from src.utils import get_device, load_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Severity head on a trained backbone (E25).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--checkpoint", default="outputs/resnet18_color_strong_p70_224_best.pth")
    parser.add_argument("--ratios", default="outputs/lesion_ratios.json")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap each split, for smoke tests")
    parser.add_argument("--out", default="outputs/severity_head.json")
    return parser.parse_args()


@torch.no_grad()
def extract(backbone, items, transform, cfg, device):
    loader = DataLoader(ItemDataset(items, transform), batch_size=cfg.data.batch_size,
                        num_workers=cfg.data.num_workers, pin_memory=True)
    feats = [backbone(x.to(device)).flatten(1).cpu() for x, _ in loader]
    return torch.cat(feats)


def fit_head(width, train, val, epochs, lr, device):
    """Sigmoid output against sqrt(ratio); both live in [0, 1]."""
    xt, yt = train[0].to(device), train[1].to(device)
    xv, yv = val[0].to(device), val[1].to(device)
    head = nn.Linear(width, 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best, best_state = float("inf"), None
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), 1024):
            idx = perm[i:i + 1024]
            opt.zero_grad()
            loss = nn.functional.mse_loss(head(xt[idx]).squeeze(1).sigmoid(), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(head(xv).squeeze(1).sigmoid(), yv))
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, best


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
    train_idx, val_idx, test_idx = splits_from_config(cfg, base)

    def gather(indices):
        """Keep only images with a cached official-mask ratio."""
        items, target = [], []
        for i in indices:
            path, label = base.samples[i]
            r = ratios.get(Path(path).name)
            if r is not None:
                items.append((path, label))
                target.append(r)
            if args.limit and len(items) >= args.limit:
                break
        return items, np.asarray(target, dtype=np.float32)

    splits = {name: gather(idx) for name, idx in
              [("train", train_idx), ("val", val_idx), ("test", test_idx)]}
    for name, (items, _) in splits.items():
        print(f"  {name:5s} {len(items):,} images")

    model = build_model(args.model, len(classes), pretrained=False)
    model.load_state_dict(load_checkpoint(args.checkpoint, map_location="cpu")["model"])
    width = strip_head(model)
    backbone = model.to(device).eval()
    print(f"backbone {args.checkpoint} -> {width}-d features")

    transform = build_transforms(cfg.data.image_size, train=False)
    cache = {name: (extract(backbone, items, transform, cfg, device),
                    torch.from_numpy(np.sqrt(target)))
             for name, (items, target) in splits.items()}
    del backbone
    torch.cuda.empty_cache()

    head, val_mse = fit_head(width, cache["train"], cache["val"], args.epochs, args.lr, device)
    head_params = sum(p.numel() for p in head.parameters())
    print(f"severity head: {head_params:,} parameters, best val MSE {val_mse:.5f}")

    # Score everything on the same test split, so floor and ceiling are comparable.
    items, official = splits["test"]
    with torch.no_grad():
        predicted = (head(cache["test"][0].to(device)).squeeze(1).sigmoid() ** 2).cpu().numpy()
    diseased = np.array([0 if "healthy" in classes[l].lower() else 1 for _, l in items])

    print("computing the Otsu baseline on the same images...")
    otsu = np.array([lesion_ratio(cv2.imread(p)) for p, _ in items], dtype=np.float32)

    # Healthy-vs-diseased AUC is largely answerable from class identity alone, and a
    # head reading classifier features can do that without measuring the leaf in
    # front of it. Correlating inside each class removes that route, so this is the
    # number that says whether severity is being estimated at all.
    labels = np.array([l for _, l in items])

    def within_class(pred):
        rhos, sizes = [], []
        for label in np.unique(labels):
            idx = np.flatnonzero(labels == label)
            if len(idx) < 30:
                continue
            rho = spearmanr(pred[idx], official[idx]).statistic
            if np.isfinite(rho):
                rhos.append(rho)
                sizes.append(len(idx))
        return round(float(np.average(rhos, weights=sizes)), 4), len(rhos)

    rho_head, n_scored = within_class(predicted)
    rho_otsu, _ = within_class(otsu)

    summary = {
        "checkpoint": args.checkpoint,
        "feature_dim": width,
        "head_parameters": head_params,
        "n_test": len(items),
        "within_class_rho_head": rho_head,
        "within_class_rho_otsu": rho_otsu,
        "within_class_n_classes": n_scored,
        "val_mse_sqrt_ratio": round(val_mse, 6),
        "auc_predicted": round(float(roc_auc_score(diseased, predicted)), 4),
        "auc_official_mask": round(float(roc_auc_score(diseased, official)), 4),
        "auc_otsu_mask": round(float(roc_auc_score(diseased, otsu)), 4),
        "spearman_vs_official": round(float(spearmanr(predicted, official).statistic), 4),
        "spearman_otsu_vs_official": round(float(spearmanr(otsu, official).statistic), 4),
        "mae_vs_official": round(float(np.abs(predicted - official).mean()), 4),
        "mae_otsu_vs_official": round(float(np.abs(otsu - official).mean()), 4),
    }
    print(f"\n  within-class rho: head {rho_head:.4f}   Otsu {rho_otsu:.4f}   "
          f"({n_scored} classes)   <- does it measure THIS leaf?")
    print(f"\n  AUC  predicted head {summary['auc_predicted']:.4f}"
          f"   <- confounded: class identity alone answers most of this")
    print(f"  AUC  Otsu (floor)   {summary['auc_otsu_mask']:.4f}")
    print(f"  AUC  official mask  {summary['auc_official_mask']:.4f}   "
          f"<- ceiling, reads masks the deployed system will not have")
    print(f"  rho vs official: head {summary['spearman_vs_official']:.4f}   "
          f"Otsu {summary['spearman_otsu_vs_official']:.4f}")
    print(f"  MAE vs official: head {summary['mae_vs_official']:.4f}   "
          f"Otsu {summary['mae_otsu_vs_official']:.4f}")

    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
