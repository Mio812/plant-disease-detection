"""Dataset loading, transforms, and reproducible train/val/test splits."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size, train):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class TransformSubset(Dataset):
    """Wrap a Subset so that train/val/test can use different transforms."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        return self.transform(image), label


def parse_class_name(name):
    """Split a PlantVillage folder name into (crop, condition, is_healthy)."""
    crop, _, condition = name.partition("___")
    condition = condition or "healthy"
    return crop, condition, "healthy" in condition.lower()


def get_dataloaders(cfg):
    base = ImageFolder(cfg.data.root)
    n_total = len(base)
    n_test = int(n_total * cfg.data.test_split)
    n_val = int(n_total * cfg.data.val_split)

    generator = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    train_tf = build_transforms(cfg.data.image_size, train=True)
    eval_tf = build_transforms(cfg.data.image_size, train=False)
    datasets = {
        "train": TransformSubset(Subset(base, train_idx), train_tf),
        "val": TransformSubset(Subset(base, val_idx), eval_tf),
        "test": TransformSubset(Subset(base, test_idx), eval_tf),
    }

    loaders = {
        split: DataLoader(
            ds,
            batch_size=cfg.data.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.data.num_workers,
            pin_memory=True,
        )
        for split, ds in datasets.items()
    }
    return loaders, base.classes


def class_distribution(root):
    """Return {class_name: image_count} for an ImageFolder-style directory."""
    root = Path(root)
    counts = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        counts[class_dir.name] = sum(1 for _ in class_dir.glob("*.*"))
    return counts
