"""Model layer: architectures and the soft-voting ensemble."""

from .ensemble import combine, ensemble_predict, member_probs, tune_weights
from .factory import freeze_backbone, trainable_parameters, CustomCNN, build_model

__all__ = ["CustomCNN", "build_model", "freeze_backbone", "trainable_parameters", "combine", "ensemble_predict",
           "member_probs", "tune_weights"]
