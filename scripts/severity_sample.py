"""Sample diseased leaves for manual severity grading.

Writes a CSV with the estimator's output and an empty ``manual_grade`` column.
Grade each row 0-3 (healthy / mild / moderate / severe), then score the
estimator with ``scripts.severity_validate``.

Usage:
    python -m scripts.severity_sample --n 150
"""

import argparse
import csv
import random
from pathlib import Path

import cv2
from torchvision.datasets import ImageFolder

from src.config import Config
from src.severity import estimate_severity


def parse_args():
    parser = argparse.ArgumentParser(description="Sample leaves for severity annotation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--out", default="outputs/severity_annotations.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    base = ImageFolder(cfg.data.root)
    diseased = [i for i, (_, lab) in enumerate(base.samples)
                if "healthy" not in base.classes[lab].lower()]
    random.seed(cfg.seed)
    chosen = random.sample(diseased, min(args.n, len(diseased)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "class", "lesion_ratio", "estimated_level", "manual_grade"])
        for i in chosen:
            path, label = base.samples[i]
            level, ratio = estimate_severity(cv2.imread(path), cfg.severity.thresholds,
                                             cfg.severity.levels)
            writer.writerow([path, base.classes[label], f"{ratio:.4f}", level, ""])
    print(f"Wrote {len(chosen)} rows to {out.as_posix()} - fill in `manual_grade` (0-3).")


if __name__ == "__main__":
    main()
