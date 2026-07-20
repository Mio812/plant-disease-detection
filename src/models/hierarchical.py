"""Factorised crop-then-disease classifier.

Field accuracy decomposes as crop identification x diagnosis given the crop, and
the crop term is the weaker of the two (46% vs 52% on PlantDoc). A flat 38-way
softmax leaves species recognition implicit. This head makes it an explicit
14-way problem and restricts the diagnosis to diseases the predicted crop can
actually have, so probability mass never leaks to another species.

    p(class) = p(crop) * p(class | crop)

forward() returns log-probabilities, not logits, so callers must exponentiate
rather than take a softmax. Modules carry ``returns_log_probs`` to signal this.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .factory import build_model

MASKED = -1e9


def crop_of(classes):
    """Index each class to its crop, from the 'Crop___Disease' naming convention."""
    crops = sorted({c.split("___")[0] for c in classes})
    lookup = {c: i for i, c in enumerate(crops)}
    return torch.tensor([lookup[c.split("___")[0]] for c in classes]), crops


def _strip_head(name, pretrained):
    """Return the backbone as a feature extractor plus its feature dimension."""
    model = build_model(name, 1, pretrained=pretrained)
    if hasattr(model, "fc"):
        dim = model.fc.in_features
        model.fc = nn.Identity()
    else:
        dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
    return model, dim


class HierarchicalClassifier(nn.Module):
    def __init__(self, name, crop_index, pretrained=True):
        super().__init__()
        self.returns_log_probs = True
        self.name = name
        crop_index = torch.as_tensor(crop_index, dtype=torch.long)
        num_classes = len(crop_index)
        num_crops = int(crop_index.max()) + 1

        self.backbone, dim = _strip_head(name, pretrained)
        self.crop_head = nn.Linear(dim, num_crops)
        self.disease_head = nn.Linear(dim, num_classes)

        # Buffers, so a checkpoint alone is enough to rebuild the model.
        self.register_buffer("crop_index", crop_index)
        mask = torch.zeros(num_crops, num_classes, dtype=torch.bool)
        mask[crop_index, torch.arange(num_classes)] = True
        self.register_buffer("class_mask", mask)
        self.register_buffer("positions", torch.arange(num_classes))

    def forward(self, x):
        features = self.backbone(x)
        crop_logp = F.log_softmax(self.crop_head(features), dim=1)

        # Normalise the disease logits within each crop, then read off the entry
        # for each class under its own crop.
        disease = self.disease_head(features).unsqueeze(1)
        disease = disease.masked_fill(~self.class_mask, MASKED)
        within = disease - disease.logsumexp(2, keepdim=True)
        within = within[:, self.crop_index, self.positions]

        return crop_logp[:, self.crop_index] + within

    @torch.no_grad()
    def crop_logits(self, x):
        return self.crop_head(self.backbone(x))


class LogProbLoss(nn.Module):
    """Negative log-likelihood with label smoothing, for models that emit log-probs."""

    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, log_p, target):
        nll = -log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        if self.smoothing:
            nll = (1.0 - self.smoothing) * nll - self.smoothing * log_p.mean(1)
        return nll.mean()


def hierarchical_from_state(state, name, pretrained=False):
    """Rebuild from a checkpoint: the crop map travels with the weights."""
    return HierarchicalClassifier(name, state["crop_index"], pretrained=pretrained)


def is_hierarchical(state):
    return "crop_head.weight" in state
