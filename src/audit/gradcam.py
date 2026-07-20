"""Grad-CAM maps, and the share of CAM mass falling inside the leaf mask.

Compared against the leaf's share of image area as a baseline.
"""

import torch
import torch.nn.functional as F


def target_layer(model, name):
    """Last convolutional stage of a supported backbone."""
    if name.startswith("resnet"):
        return model.layer4
    if name in ("mobilenet_v2", "efficientnet_b0"):
        return model.features
    if name == "custom_cnn":
        return model.features
    raise ValueError(f"no Grad-CAM layer configured for {name!r}")


def grad_cam(model, layer, images, device):
    """Normalised Grad-CAM maps for each image's predicted class."""
    store = {}
    handles = [
        layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o)),
        layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0])),
    ]
    images = images.to(device)
    logits = model(images)
    predicted = logits.argmax(1)
    model.zero_grad()
    logits[torch.arange(len(predicted)), predicted].sum().backward()
    for handle in handles:
        handle.remove()

    weights = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = (weights * store["a"]).sum(1).relu()
    cam = F.interpolate(cam.unsqueeze(1), size=images.shape[-2:], mode="bilinear",
                        align_corners=False).squeeze(1)
    flat = cam.flatten(1)
    flat = flat - flat.min(1, keepdim=True).values
    flat = flat / flat.max(1, keepdim=True).values.clamp_min(1e-8)
    return flat.view_as(cam).detach().cpu().numpy(), predicted.detach().cpu()


def leaf_attention(cam, leaf_mask):
    """(CAM mass inside the leaf, leaf share of image area)."""
    total = float(cam.sum())
    inside = float((cam * leaf_mask).sum()) / max(total, 1e-8)
    return inside, float(leaf_mask.mean())
