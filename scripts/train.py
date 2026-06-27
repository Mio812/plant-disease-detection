"""Train a classifier defined by a YAML config.

Usage:
    python -m scripts.train --config configs/default.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.config import Config
from src.data import get_dataloaders
from src.engine import evaluate, fit
from src.metrics import full_report
from src.models import build_model
from src.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a plant-disease classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, help="Override model.name from the config.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    if args.model:
        cfg.model.name = args.model
    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    loaders, classes = get_dataloaders(cfg)
    print(
        f"Classes: {len(classes)} | "
        f"train={len(loaders['train'].dataset)} "
        f"val={len(loaders['val'].dataset)} "
        f"test={len(loaders['test'].dataset)}"
    )

    model = build_model(cfg.model.name, len(classes), cfg.model.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{cfg.model.name}_best.pth"

    history = fit(
        model,
        loaders,
        criterion,
        optimizer,
        scheduler,
        device,
        cfg.train.epochs,
        cfg.train.early_stop_patience,
        ckpt_path,
    )

    test_metrics, y_true, y_pred = evaluate(model, loaders["test"], criterion, device)
    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("\n" + full_report(y_true, y_pred, classes))

    with open(out_dir / f"{cfg.model.name}_history.json", "w", encoding="utf-8") as f:
        json.dump(
            {"history": history, "test_metrics": test_metrics, "classes": classes}, f, indent=2
        )


if __name__ == "__main__":
    main()
