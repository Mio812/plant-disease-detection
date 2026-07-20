"""One place to load a checkpoint and run inference.

Every evaluation path -- PlantVillage, segmented, PlantDoc, Grad-CAM -- used to
carry its own copy of this loop, which is how they drift apart.
"""

import torch
import torch.nn.functional as F

from ..models.factory import build_model
from ..utils import load_checkpoint


def load_model(name, num_classes, checkpoint, device, pretrained=False):
    """Build ``name`` and restore ``checkpoint`` in eval mode."""
    model = build_model(name, num_classes, pretrained=pretrained).to(device)
    model.load_state_dict(load_checkpoint(checkpoint, map_location=device)["model"])
    model.eval()
    return model


def enable_batchnorm_adaptation(model):
    """AdaBN: let BatchNorm use the target batch statistics instead of the stored ones."""
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train()
    return model


@torch.no_grad()
def predict_loader(model, loader, device, tta=False):
    """Softmax probabilities over a DataLoader, with the labels it yielded."""
    probs, targets = [], []
    for images, labels in loader:
        images = images.to(device)
        p = F.softmax(model(images), dim=1)
        if tta:
            p = (p + F.softmax(model(torch.flip(images, [3])), dim=1)) / 2
        probs.append(p.cpu())
        targets.extend(labels.tolist())
    return torch.cat(probs), targets


@torch.no_grad()
def predict_tensor(model, images, device, batch_size=64, tta=False):
    """Softmax probabilities for an in-memory image batch."""
    out = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        p = F.softmax(model(batch), dim=1)
        if tta:
            p = (p + F.softmax(model(torch.flip(batch, [3])), dim=1)) / 2
        out.append(p.cpu())
    return torch.cat(out)
