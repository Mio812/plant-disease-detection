"""Objective check of the severity estimator that needs no manual grades.

If the lesion-area ratio measures anything real, it must be near zero on healthy
leaves and larger on diseased ones. This script scores that separation (ROC-AUC)
and compares two leaf-segmentation sources: Otsu thresholding versus
PlantVillage's official ``segmented`` masks.

Usage:
    python -m scripts.severity_probe --n 150
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
from torchvision.datasets import ImageFolder

from src.bias import name_key, segmented_index
from src.config import Config
from src.data import variant_root
from src.severity import leaf_mask_from_segmented, lesion_ratio, severity_level


def parse_args():
    parser = argparse.ArgumentParser(description="Validate severity without manual labels.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n", type=int, default=150, help="images per group")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    base = ImageFolder(cfg.data.root)
    seg_root = variant_root(cfg.data.root, "segmented")

    healthy_idx, diseased_idx = [], []
    for i, (_, label) in enumerate(base.samples):
        (healthy_idx if "healthy" in base.classes[label].lower() else diseased_idx).append(i)
    rng = random.Random(cfg.seed)
    chosen = ([(i, 0) for i in rng.sample(healthy_idx, min(args.n, len(healthy_idx)))] +
              [(i, 1) for i in rng.sample(diseased_idx, min(args.n, len(diseased_idx)))])

    index, rows = {}, []
    for i, is_diseased in chosen:
        path, label = base.samples[i]
        image = cv2.imread(path)
        if image is None:
            continue
        class_name = base.classes[label]
        if class_name not in index:
            index[class_name] = segmented_index(seg_root, class_name)
        twin = index[class_name].get(name_key(Path(path).name))
        official = None
        if twin is not None:
            seg = cv2.imread(twin)
            if seg is not None:
                official = lesion_ratio(image, leaf_mask_from_segmented(cv2.resize(seg, image.shape[1::-1])))
        rows.append((is_diseased, lesion_ratio(image), official))

    y = np.array([r[0] for r in rows])
    otsu = np.array([r[1] for r in rows])
    have = np.array([r[2] is not None for r in rows])
    official = np.array([r[2] if r[2] is not None else 0.0 for r in rows])

    metrics = {
        "n": int(len(rows)),
        "auc_otsu_mask": float(roc_auc_score(y, otsu)),
        "auc_official_mask": float(roc_auc_score(y[have], official[have])),
        "mean_ratio_healthy_otsu": float(otsu[y == 0].mean()),
        "mean_ratio_diseased_otsu": float(otsu[y == 1].mean()),
        "mean_ratio_healthy_official": float(official[have & (y == 0)].mean()),
        "mean_ratio_diseased_official": float(official[have & (y == 1)].mean()),
    }
    levels, thresholds = list(cfg.severity.levels), list(cfg.severity.thresholds)
    grades = [severity_level(r, thresholds, levels) for r in official[have & (y == 1)]]
    metrics["diseased_grade_distribution"] = {lv: grades.count(lv) for lv in levels}

    print(f"n = {metrics['n']} ({int((y == 0).sum())} healthy / {int((y == 1).sum())} diseased)")
    print(f"  ROC-AUC healthy vs diseased, Otsu mask     : {metrics['auc_otsu_mask']:.3f}")
    print(f"  ROC-AUC healthy vs diseased, official mask : {metrics['auc_official_mask']:.3f}")
    print(f"  mean lesion ratio  healthy {metrics['mean_ratio_healthy_official']:.3f} "
          f"| diseased {metrics['mean_ratio_diseased_official']:.3f}  (official mask)")
    print(f"  diseased grades: {metrics['diseased_grade_distribution']}")

    out = Path(cfg.output_dir) / "severity_probe.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved {out.as_posix()}")


if __name__ == "__main__":
    main()
