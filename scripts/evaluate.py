"""Evaluate a trained checkpoint on the test split and save reports.

Writes a normalised confusion-matrix plot, the per-class classification report,
and the exact off-diagonal confusion pairs (true -> pred counts).

Usage:
    python -m scripts.evaluate --config configs/default.yaml --checkpoint outputs/resnet18_best.pth
"""

import argparse
import json
from pathlib import Path

import torch.nn as nn

from src.config import Config
from src.data import get_dataloaders
from src.engine import evaluate
from src.metrics import confusion, full_report
from src.models import build_model
from src.utils import get_device, load_checkpoint, set_seed
from src.visualize import plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None, help="Override model.name from the config.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    if args.model:
        cfg.model.name = args.model
    set_seed(cfg.seed)
    device = get_device()

    loaders, classes = get_dataloaders(cfg)
    model = build_model(cfg.model.name, len(classes), pretrained=False).to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint, map_location=device)["model"])

    criterion = nn.CrossEntropyLoss()
    metrics, y_true, y_pred = evaluate(model, loaders["test"], criterion, device)
    report = full_report(y_true, y_pred, classes)
    print("Test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("\n" + report)

    out_dir = Path(cfg.output_dir)
    cm = confusion(y_true, y_pred, len(classes))
    pairs = [
        {"true": classes[i], "pred": classes[j], "count": int(cm[i, j])}
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(key=lambda p: p["count"], reverse=True)

    (out_dir / f"{cfg.model.name}_classification_report.txt").write_text(report, encoding="utf-8")
    with open(out_dir / f"{cfg.model.name}_confused_pairs.json", "w", encoding="utf-8") as f:
        json.dump({"total_errors": sum(p["count"] for p in pairs), "pairs": pairs}, f, indent=2)

    if pairs:
        print("\nTop confused pairs (true -> pred):")
        for p in pairs[:15]:
            print(f"  {p['count']:3d}  {p['true']} -> {p['pred']}")

    fig = plot_confusion_matrix(cm, classes, normalize=True)
    fig_path = out_dir / f"{cfg.model.name}_confusion_matrix.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nSaved confusion matrix to {fig_path}")


if __name__ == "__main__":
    main()
