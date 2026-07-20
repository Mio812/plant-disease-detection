"""Model layer: architectures and the soft-voting ensemble."""

from .ensemble import combine, ensemble_predict, member_probs, tune_weights
from .hierarchical import (
    HierarchicalClassifier, LogProbLoss, crop_of, hierarchical_from_state, is_hierarchical,
)
from .factory import freeze_backbone, trainable_parameters, CustomCNN, build_model

__all__ = ["CustomCNN", "build_model", "HierarchicalClassifier", "LogProbLoss", "crop_of",
           "hierarchical_from_state", "is_hierarchical", "freeze_backbone", "trainable_parameters", "combine", "ensemble_predict",
           "member_probs", "tune_weights"]
