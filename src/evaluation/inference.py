"""Checkpoint loading and inference, shared by every evaluation path."""

import torch
import torch.nn.functional as F

from ..models.factory import build_model
from ..models.hierarchical import hierarchical_from_state, is_hierarchical
from ..utils import load_checkpoint


def to_probs(model, outputs):
    """Hierarchical models already emit log-probabilities; everything else emits logits."""
    return outputs.exp() if getattr(model, "returns_log_probs", False) else F.softmax(outputs, dim=1)


def load_model(name, num_classes, checkpoint, device, pretrained=False):
    """Build ``name`` and restore ``checkpoint``, in eval mode."""
    state = load_checkpoint(checkpoint, map_location=device)["model"]
    if is_hierarchical(state):
        model = hierarchical_from_state(state, name, pretrained=False)
    else:
        model = build_model(name, num_classes, pretrained=pretrained)
    model = model.to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def enable_batchnorm_adaptation(model):
    """AdaBN: BatchNorm uses target-batch statistics instead of the stored ones."""
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train()
    return model


@torch.no_grad()
def predict_loader(model, loader, device, tta=False):
    """Softmax probabilities and labels over a DataLoader."""
    probs, targets = [], []
    for images, labels in loader:
        images = images.to(device)
        p = to_probs(model, model(images))
        if tta:
            p = (p + to_probs(model, model(torch.flip(images, [3])))) / 2
        probs.append(p.cpu())
        targets.extend(labels.tolist())
    return torch.cat(probs), targets


@torch.no_grad()
def predict_tensor(model, images, device, batch_size=64, tta=False):
    """Softmax probabilities for an in-memory batch."""
    out = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        p = to_probs(model, model(batch))
        if tta:
            p = (p + to_probs(model, model(torch.flip(batch, [3])))) / 2
        out.append(p.cpu())
    return torch.cat(out)
