"""Train/validation/test split shared by every experiment.

PlantVillage photographs each leaf many times, so a random split leaks near-duplicate
images of one leaf across train and test, inflating the held-out score. The split
keeps every image of a leaf (from ``leaf-map.json``) on one side, partitioning leaves
within each class so it stays stratified. An image with no recorded leaf is its own
singleton -- with no known duplicate it cannot leak.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .variants import name_key


def make_splits(seed, n_total, val_split, test_split):
    """Random split by index. Kept for reference; leaks leaf duplicates, so it is
    not the default path -- see ``leaf_grouped_splits``."""
    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed)).tolist()
    return perm[n_test + n_val:], perm[n_test:n_test + n_val], perm[:n_test]


def leaf_map_path(color_root):
    return Path(color_root).parent.parent / "leaf-map.json"


def leaf_groups(samples, labels, color_root):
    """Group id per sample: its leaf, or a unique singleton. Scoped by class label,
    since ``name_key`` can collide across classes and would otherwise merge leaves."""
    path = leaf_map_path(color_root)
    leafmap = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    groups = []
    for i, ((sample_path, _), label) in enumerate(zip(samples, labels)):
        entry = leafmap.get(name_key(Path(sample_path).name).lower())
        groups.append(f"{label}:::{entry[0]}" if entry else f"__singleton_{i}")
    return groups


def leaf_grouped_splits(seed, groups, labels, val_split, test_split):
    """Partition each class's leaves, then place every image with its leaf, so all
    classes stay represented and no leaf spans the partition."""
    rng = np.random.default_rng(seed)
    leaves_of_class = defaultdict(set)
    indices_of_leaf = defaultdict(list)
    label_of_leaf = {}
    for i, (group, label) in enumerate(zip(groups, labels)):
        leaves_of_class[label].add(group)
        indices_of_leaf[group].append(i)
        # A leaf must belong to exactly one class, or per-class partitioning is unsound.
        if label_of_leaf.setdefault(group, label) != label:
            raise ValueError(f"leaf {group!r} spans more than one class")

    train, val, test = [], [], []
    for label, leaves in leaves_of_class.items():
        ordered = sorted(leaves)
        rng.shuffle(ordered)
        n = len(ordered)
        if n == 1:
            n_test = n_val = 0            # degenerate: one leaf cannot be held out honestly
        elif n == 2:
            n_test, n_val = 1, 0
        else:
            n_test = max(1, round(n * test_split))
            n_val = max(1, round(n * val_split))
        for leaf in ordered[:n_test]:
            test += indices_of_leaf[leaf]
        for leaf in ordered[n_test:n_test + n_val]:
            val += indices_of_leaf[leaf]
        for leaf in ordered[n_test + n_val:]:
            train += indices_of_leaf[leaf]
    return sorted(train), sorted(val), sorted(test)


def splits_from_config(cfg, base):
    """Leaf-grouped split for an ImageFolder. ``base`` may be the ImageFolder or,
    for callers that still pass a count, an int -- which forces the random split
    and is only kept working so nothing breaks silently."""
    if isinstance(base, int):
        return make_splits(cfg.seed, base, cfg.data.val_split, cfg.data.test_split)
    labels = [label for _, label in base.samples]
    groups = leaf_groups(base.samples, labels, cfg.data.root)
    return leaf_grouped_splits(cfg.seed, groups, labels, cfg.data.val_split, cfg.data.test_split)
