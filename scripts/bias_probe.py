"""Background-pixel probe: predict the class from border pixels alone.

If PlantVillage carried no capture bias, a classifier trained on a handful of
background pixels would score near chance (1/38 = 2.6%). It does not.

Usage:
    python -m scripts.bias_probe --config configs/default.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torchvision.datasets import ImageFolder

from src.bias import BORDER_POSITIONS, border_features
from src.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Background-pixel bias probe.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--per-class-train", type=int, default=100)
    parser.add_argument("--per-class-test", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    base = ImageFolder(cfg.data.root)
    n = len(base)
    n_test = int(n * cfg.data.test_split)
    n_val = int(n * cfg.data.val_split)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed)).tolist()

    quota, rows = {}, []
    for split, indices, cap in (("train", perm[n_test + n_val:], args.per_class_train),
                                ("test", perm[:n_test], args.per_class_test)):
        for i in indices:
            label = base.targets[i]
            if quota.get((split, label), 0) < cap:
                quota[(split, label)] = quota.get((split, label), 0) + 1
                rows.append((base.samples[i][0], label, split))

    X = np.array([border_features(p) for p, _, _ in rows], dtype=np.float32)
    y = np.array([lab for _, lab, _ in rows])
    is_train = np.array([s == "train" for _, _, s in rows])

    scaler = StandardScaler().fit(X[is_train])
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X[is_train]), y[is_train])
    accuracy = float(clf.score(scaler.transform(X[~is_train]), y[~is_train]))

    print(f"Border pixels used: {len(BORDER_POSITIONS)}")
    print(f"train={is_train.sum()} test={(~is_train).sum()} classes={len(base.classes)}")
    print(f"TEST ACCURACY = {accuracy * 100:.1f}%  (random = {100 / len(base.classes):.1f}%)")

    out = Path(cfg.output_dir) / "bias_probe.json"
    out.write_text(json.dumps({"n_pixels": len(BORDER_POSITIONS), "test_accuracy": accuracy,
                               "random": 1 / len(base.classes)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
