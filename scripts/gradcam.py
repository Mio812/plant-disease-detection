"""Grad-CAM explanations, and how much attention actually lands on the leaf.

Beyond the usual heat-map picture this reports a quantitative audit metric: the
share of Grad-CAM mass falling inside the official leaf mask, compared with the
leaf's share of the image area. Attention no better than the area baseline means
the model is not preferentially looking at the leaf.

Usage:
    python -m scripts.gradcam --model resnet18 --n 60
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import ImageFolder

from src.bias import name_key, segmented_index
from src.config import Config
from src.data import build_transforms, variant_root
from src.models import build_model
from src.utils import get_device, load_checkpoint, set_seed


def target_layer(model, name):
    if name.startswith("resnet"):
        return model.layer4
    if name in ("mobilenet_v2", "efficientnet_b0"):
        return model.features
    raise ValueError(f"no Grad-CAM layer configured for {name!r}")


def grad_cam(model, layer, images, device):
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))
    images = images.to(device).requires_grad_(False)
    logits = model(images)
    predicted = logits.argmax(1)
    model.zero_grad()
    logits[torch.arange(len(predicted)), predicted].sum().backward()
    h1.remove()
    h2.remove()
    weights = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = (weights * store["a"]).sum(1).relu()
    cam = F.interpolate(cam.unsqueeze(1), size=images.shape[-2:], mode="bilinear",
                        align_corners=False).squeeze(1)
    flat = cam.flatten(1)
    flat = flat - flat.min(1, keepdim=True).values
    flat = flat / flat.max(1, keepdim=True).values.clamp_min(1e-8)
    return flat.view_as(cam).detach().cpu().numpy(), predicted.cpu()


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM and leaf-attention audit.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--figure", default="outputs/gradcam.png")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    device = get_device()
    checkpoint = args.checkpoint or f"{cfg.output_dir}/{args.model}_best.pth"

    base = ImageFolder(cfg.data.root)
    seg_root = variant_root(cfg.data.root, "segmented")
    model = build_model(args.model, len(base.classes), pretrained=False).to(device)
    model.load_state_dict(load_checkpoint(checkpoint, map_location=device)["model"])
    model.eval()

    rng = random.Random(cfg.seed)
    diseased = [i for i, (_, l) in enumerate(base.samples)
                if "healthy" not in base.classes[l].lower()]
    chosen = rng.sample(diseased, min(args.n, len(diseased)))
    transform = build_transforms(cfg.data.image_size, train=False)

    index, records, gallery = {}, [], []
    for start in range(0, len(chosen), 16):
        batch = chosen[start:start + 16]
        tensors, masks, originals = [], [], []
        for i in batch:
            path, label = base.samples[i]
            class_name = base.classes[label]
            if class_name not in index:
                index[class_name] = segmented_index(seg_root, class_name)
            twin = index[class_name].get(name_key(Path(path).name))
            if twin is None:
                continue
            seg = cv2.imread(twin)
            if seg is None:
                continue
            size = (cfg.data.image_size, cfg.data.image_size)
            masks.append(cv2.resize(seg, size).sum(axis=2) > 25)
            image = Image.open(path).convert("RGB")
            tensors.append(transform(image))
            originals.append(np.array(image.resize(size)))
        if not tensors:
            continue
        cams, _ = grad_cam(model, target_layer(model, args.model), torch.stack(tensors), device)
        for cam, mask, original in zip(cams, masks, originals):
            total = cam.sum()
            records.append({"attention_in_leaf": float((cam * mask).sum() / max(total, 1e-8)),
                            "leaf_area_fraction": float(mask.mean())})
            if len(gallery) < 4:
                gallery.append((original, cam, mask))

    attention = np.array([r["attention_in_leaf"] for r in records])
    area = np.array([r["leaf_area_fraction"] for r in records])
    summary = {"model": args.model, "n": int(len(records)),
               "mean_attention_in_leaf": float(attention.mean()),
               "mean_leaf_area_fraction": float(area.mean()),
               "attention_lift_over_area": float(attention.mean() - area.mean())}
    print(f"{args.model}: n={summary['n']}")
    print(f"  Grad-CAM mass inside the leaf : {summary['mean_attention_in_leaf'] * 100:.1f}%")
    print(f"  leaf share of image area      : {summary['mean_leaf_area_fraction'] * 100:.1f}%")
    print(f"  lift over the area baseline   : {summary['attention_lift_over_area'] * 100:+.1f} points")
    Path(f"{cfg.output_dir}/gradcam_audit.json").write_text(json.dumps(summary, indent=2),
                                                           encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(gallery), figsize=(3 * len(gallery), 6))
    for col, (original, cam, mask) in enumerate(gallery):
        axes[0, col].imshow(original)
        axes[0, col].set_title("input", fontsize=9)
        axes[1, col].imshow(original)
        axes[1, col].imshow(cam, cmap="jet", alpha=0.5)
        axes[1, col].set_title("Grad-CAM", fontsize=9)
        for row in (0, 1):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle(f"{args.model}: {summary['mean_attention_in_leaf'] * 100:.0f}% of attention "
                 f"on the leaf (leaf covers {summary['mean_leaf_area_fraction'] * 100:.0f}% of the image)")
    fig.tight_layout()
    fig.savefig(args.figure, dpi=170)
    print(f"saved {args.figure}")


if __name__ == "__main__":
    main()
