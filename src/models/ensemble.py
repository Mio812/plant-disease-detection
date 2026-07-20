"""Soft-voting ensemble over trained baseline checkpoints.

The ensemble averages the members' softmax probabilities. Weights can be tuned
on the validation split to favour stronger members; equal weights are the default.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .factory import build_model
from ..utils import load_checkpoint


@torch.no_grad()
def member_probs(members, num_classes, loader, device):
    """Return per-member softmax tensors and the shared label list.

    ``members`` is a list of ``(model_name, checkpoint_path)``. The loader must
    be unshuffled so every member sees the samples in the same order.
    """
    probs, y_true = [], []
    for i, (name, ckpt_path) in enumerate(members):
        model = build_model(name, num_classes, pretrained=False).to(device)
        model.load_state_dict(load_checkpoint(ckpt_path, map_location=device)["model"])
        model.eval()
        batches = []
        for images, targets in loader:
            batches.append(F.softmax(model(images.to(device)), dim=1).cpu())
            if i == 0:
                y_true.extend(targets.tolist())
        probs.append(torch.cat(batches))
    return probs, y_true


def combine(probs, weights=None):
    """Weighted average of member probability tensors (equal weights by default)."""
    weights = [1.0] * len(probs) if weights is None else list(weights)
    return sum(w * p for w, p in zip(weights, probs)) / sum(weights)


def tune_weights(probs, y_true, step=0.05):
    """Grid-search simplex weights (three members) that maximise accuracy."""
    y = np.array(y_true)
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    best_w, best_acc = None, -1.0
    for w1 in grid:
        for w2 in grid[grid <= 1.0 - w1 + 1e-9]:
            w = (float(w1), float(w2), float(1.0 - w1 - w2))
            acc = (combine(probs, w).argmax(1).numpy() == y).mean()
            if acc > best_acc:
                best_acc, best_w = acc, tuple(round(x, 3) for x in w)
    return best_w


@torch.no_grad()
def ensemble_predict(members, num_classes, loader, device, weights=None):
    """Convenience wrapper: member probabilities combined into (probs, y_true)."""
    probs, y_true = member_probs(members, num_classes, loader, device)
    return combine(probs, weights), y_true
