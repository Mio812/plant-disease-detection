"""Quantify residual leakage the leaf-map missed -- calibrated against hard negatives.

The leaf-grouped split removes every known same-leaf duplicate, but 24% of images
have no leaf-map entry and are treated as singletons. If two of those are secretly
the same leaf, a little leakage survives.

A frozen ImageNet embedding clusters by class, so "same leaf" and "different leaf,
same class" both score high cosine -- the discriminating question is whether a
test image is MORE similar to a train image than different leaves of the same class
ever are. Calibration therefore uses two distributions: same-leaf pairs (positive)
and different-leaf-same-class pairs (hard negative). The threshold is a high
percentile of the negatives; a test image above it is more alike than same-class
membership explains, i.e. a probable missed duplicate. If the two distributions do
not separate, the method cannot bound residual leakage and says so.

Usage:
    python -m scripts.near_dup_check
    python -m scripts.near_dup_check --limit 400
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder

from src.config import Config
from src.data import ItemDataset, build_transforms, splits_from_config
from src.data.splits import leaf_groups
from src.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Residual-leakage near-duplicate check.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--neg-percentile", type=float, default=99.9,
                   help="a test image more similar than this percentile of "
                        "different-leaf-same-class pairs counts as a probable duplicate")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="outputs/residual_leakage.json")
    return p.parse_args()


@torch.no_grad()
def embed(indices, base, transform, cfg, backbone, device):
    loader = DataLoader(ItemDataset([base.samples[i] for i in indices], transform),
                        batch_size=128, num_workers=cfg.data.num_workers, pin_memory=True)
    out = []
    for x, _ in loader:
        out.append(nn.functional.normalize(backbone(x.to(device)).flatten(1), dim=1).cpu())
    return torch.cat(out)


def sample_pair_sims(emb, groups, labels, device, same_leaf, n=40000, seed=0):
    """Cosine sims of random pairs that are either same-leaf or same-class-different-leaf."""
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for i, l in enumerate(labels):
        by_class[l].append(i)
    e = emb.to(device)
    sims, tries = [], 0
    while len(sims) < n and tries < n * 40:
        tries += 1
        c = int(rng.integers(0, max(labels) + 1))
        members = by_class.get(c, [])
        if len(members) < 2:
            continue
        i, j = rng.choice(members, 2, replace=False)
        gi, gj = groups[i], groups[j]
        is_same_leaf = (gi == gj) and not gi.startswith("__singleton")
        if is_same_leaf == same_leaf:
            sims.append(float(e[i] @ e[j]))
    return np.array(sims)


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    cfg.data.image_size = 224
    set_seed(cfg.seed)
    device = get_device()

    base = ImageFolder(cfg.data.root)
    labels = [l for _, l in base.samples]
    groups = leaf_groups(base.samples, labels, cfg.data.root)
    train_idx, _, test_idx = splits_from_config(cfg, base)
    if args.limit:
        train_idx, test_idx = train_idx[:args.limit], test_idx[:args.limit]

    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device).eval()
    transform = build_transforms(cfg.data.image_size, train=False)

    print(f"embedding {len(train_idx):,} train + {len(test_idx):,} test ...")
    train_emb = embed(train_idx, base, transform, cfg, backbone, device)
    test_emb = embed(test_idx, base, transform, cfg, backbone, device)
    train_groups = [groups[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_groups = [groups[i] for i in test_idx]

    pos = sample_pair_sims(train_emb, train_groups, train_labels, device, same_leaf=True)
    neg = sample_pair_sims(train_emb, train_groups, train_labels, device, same_leaf=False)
    print("\nsimilarity distributions (do same-leaf and same-class-different-leaf separate?)")
    for name, s in [("same leaf (positive)", pos), ("diff leaf same class (negative)", neg)]:
        print(f"  {name:34s} n={len(s):6,}  median {np.median(s):.3f}  "
              f"p10 {np.percentile(s,10):.3f}  p90 {np.percentile(s,90):.3f}")
    threshold = float(np.percentile(neg, args.neg_percentile))
    frac_pos_above = float((pos >= threshold).mean())
    print(f"\nthreshold = {args.neg_percentile} pct of negatives = {threshold:.4f}   "
          f"({100*frac_pos_above:.0f}% of true same-leaf pairs clear it)")

    # Flag test images whose nearest OUT-OF-GROUP train neighbour beats the threshold.
    tg = np.array(train_groups)
    train_dev = train_emb.to(device)
    flagged = 0
    for start in range(0, len(test_emb), 256):
        sims = test_emb[start:start + 256].to(device) @ train_dev.T
        for r in range(sims.shape[0]):
            g = test_groups[start + r]
            row = sims[r]
            if not g.startswith("__singleton"):
                row = row.clone()
                row[torch.from_numpy(tg == g).to(device)] = -1.0
            if float(row.max()) >= threshold:
                flagged += 1

    n = len(test_idx)
    summary = {
        "n_test": n,
        "threshold": round(threshold, 4),
        "neg_percentile": args.neg_percentile,
        "same_leaf_median_sim": round(float(np.median(pos)), 4),
        "diff_leaf_same_class_median_sim": round(float(np.median(neg)), 4),
        "separation": round(float(np.median(pos) - np.median(neg)), 4),
        "true_duplicate_recall_at_threshold": round(frac_pos_above, 4),
        "test_flagged": flagged,
        "test_flagged_pct": round(100.0 * flagged / n, 2),
    }
    print(f"\nresidual leakage estimate: {flagged}/{n} test images "
          f"({summary['test_flagged_pct']:.2f}%) have an out-of-group train near-duplicate")
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
