"""Robust training pipeline aimed at closing the laboratory-to-field gap.

PlantVillage backgrounds are correlated with the labels, so a model can reach
99% without looking at the leaf. Compositing each segmented leaf onto a random
background destroys that shortcut and forces the network to use leaf features.
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..data.variants import name_key, variant_index
from ..data.plantvillage import IMAGENET_MEAN, IMAGENET_STD


def random_background(size, rng):
    """Generate a random background: solid, gradient, noise or blurred blobs."""
    w, h = size
    mode = rng.randint(0, 3)
    if mode == 0:
        bg = np.full((h, w, 3), [rng.randint(0, 255) for _ in range(3)], dtype=np.uint8)
    elif mode == 1:
        top = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
        bottom = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
        ramp = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        bg = (top * (1 - ramp) + bottom * ramp).repeat(w, axis=1).astype(np.uint8)
    elif mode == 2:
        bg = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    else:
        small = np.random.randint(0, 256, (max(2, h // 32), max(2, w // 32), 3), dtype=np.uint8)
        bg = np.array(Image.fromarray(small).resize((w, h), Image.BICUBIC))
    noise = np.random.normal(0, 8, bg.shape)
    return np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def build_strong_transforms(image_size):
    """Heavier augmentation than the baseline: RandAugment, blur, erasing."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(30),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


class BackgroundRandomised(Dataset):
    """PlantVillage training images with the background replaced at random.

    ``p_random`` controls how often the background is swapped; the remainder are
    left untouched so the model still sees the original distribution.
    """

    def __init__(self, samples, classes, segmented_root, image_size, p_random=0.7, seed=0):
        self.samples = samples
        self.segmented_root = segmented_root
        self.transform = build_strong_transforms(image_size)
        self.p_random = p_random
        self.rng = random.Random(seed)
        self._index = {}
        self.classes = classes

    def _segmented(self, colour_path, class_name):
        if class_name not in self._index:
            self._index[class_name] = variant_index(self.segmented_root, class_name)
        return self._index[class_name].get(name_key(os.path.basename(colour_path)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        image = Image.open(path).convert("RGB")
        if self.rng.random() < self.p_random:
            seg_path = self._segmented(path, self.classes[label])
            if seg_path is not None:
                seg = Image.open(seg_path).convert("RGB").resize(image.size)
                mask = np.array(seg).sum(axis=2) > 25
                if mask.any():
                    bg = random_background(image.size, self.rng)
                    composed = np.where(mask[..., None], np.array(image), bg)
                    image = Image.fromarray(composed.astype(np.uint8))
        return self.transform(image), label
