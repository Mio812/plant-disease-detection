"""Reproducibility, device, and checkpoint helpers."""

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, map_location=None):
    return torch.load(path, map_location=map_location)


class AverageMeter:
    """Running average of a scalar over a stream of batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)
