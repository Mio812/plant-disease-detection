"""Evaluation layer: metrics, a single inference path, and plots."""

from .inference import (
    enable_batchnorm_adaptation,
    load_model,
    predict_loader,
    predict_tensor,
    to_probs,
)
from .metrics import compute_metrics, confusion, full_report, mcnemar
from .visualize import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_history,
    show_samples,
)

__all__ = ["load_model", "predict_loader", "to_probs", "predict_tensor", "enable_batchnorm_adaptation",
           "compute_metrics", "confusion", "full_report", "mcnemar",
           "plot_class_distribution", "plot_confusion_matrix", "plot_history", "show_samples"]
