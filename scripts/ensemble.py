"""Evaluate the soft-voting ensemble of the three baselines.

Member weights are tuned on the validation split (grid search maximising
validation accuracy), then applied to the held-out test split.

Usage:
    python -m scripts.ensemble --config configs/default.yaml
"""

import argparse
import json
from pathlib import Path

from src.config import Config
from src.data import get_dataloaders
from src.models import combine, member_probs, tune_weights
from src.evaluation import compute_metrics, confusion, full_report
from src.utils import get_device, set_seed
from src.evaluation import plot_confusion_matrix

MEMBERS = ["custom_cnn", "resnet18", "mobilenet_v2"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the baseline ensemble.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    device = get_device()
    out_dir = Path(cfg.output_dir)

    loaders, classes = get_dataloaders(cfg)
    members = [(name, out_dir / f"{name}_best.pth") for name in MEMBERS]

    val_probs, val_true = member_probs(members, len(classes), loaders["val"], device)
    weights = tune_weights(val_probs, val_true)
    print("Tuned weights:", dict(zip(MEMBERS, weights)))

    test_probs, y_true = member_probs(members, len(classes), loaders["test"], device)
    y_pred = combine(test_probs, weights).argmax(1).tolist()

    metrics = compute_metrics(y_true, y_pred)
    print("Ensemble test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    report = full_report(y_true, y_pred, classes)
    (out_dir / "ensemble_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion(y_true, y_pred, len(classes))
    pairs = [
        {"true": classes[i], "pred": classes[j], "count": int(cm[i, j])}
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(key=lambda p: p["count"], reverse=True)
    with open(out_dir / "ensemble_confused_pairs.json", "w", encoding="utf-8") as f:
        json.dump({"total_errors": sum(p["count"] for p in pairs), "pairs": pairs}, f, indent=2)
    with open(out_dir / "ensemble_history.json", "w", encoding="utf-8") as f:
        json.dump(
            {"members": MEMBERS, "weights": list(weights), "test_metrics": metrics, "classes": classes},
            f, indent=2,
        )

    plot_confusion_matrix(cm, classes, normalize=True).savefig(
        out_dir / "ensemble_confusion_matrix.png", dpi=150
    )
    print(f"Saved ensemble reports to {out_dir}")


if __name__ == "__main__":
    main()
