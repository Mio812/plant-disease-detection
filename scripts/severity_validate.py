"""Score the severity estimator against manual grades.

Reads the CSV produced by ``scripts.severity_sample`` once ``manual_grade`` has
been filled in, and reports Spearman correlation, mean absolute error and
quadratic-weighted Cohen's kappa on the ordinal levels.

Usage:
    python -m scripts.severity_validate --csv outputs/severity_annotations.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from src.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Validate severity against manual grades.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--csv", default="outputs/severity_annotations.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    levels = list(cfg.severity.levels)

    ratios, predicted, manual = [], [], []
    with open(args.csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["manual_grade"].strip():
                continue
            ratios.append(float(row["lesion_ratio"]))
            predicted.append(levels.index(row["estimated_level"]))
            manual.append(int(row["manual_grade"]))
    if not manual:
        raise SystemExit("No graded rows found - fill in the manual_grade column first.")

    predicted, manual = np.array(predicted), np.array(manual)
    rho, p = spearmanr(ratios, manual)
    metrics = {
        "n_graded": int(len(manual)),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "mae_levels": float(np.abs(predicted - manual).mean()),
        "exact_agreement": float((predicted == manual).mean()),
        "quadratic_kappa": float(cohen_kappa_score(predicted, manual, weights="quadratic")),
    }
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    Path(cfg.output_dir, "severity_validation.json").write_text(json.dumps(metrics, indent=2),
                                                                encoding="utf-8")


if __name__ == "__main__":
    main()
