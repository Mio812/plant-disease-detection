"""Training layer: optimisation loops and augmentation."""

from .augment import BackgroundRandomised, build_strong_transforms, random_background
from .engine import evaluate, fit, train_one_epoch

__all__ = ["BackgroundRandomised", "build_strong_transforms", "random_background",
           "evaluate", "fit", "train_one_epoch"]
