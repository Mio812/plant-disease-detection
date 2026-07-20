"""Classification metrics and reports."""

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def full_report(y_true, y_pred, class_names):
    # labels= is required: on small evaluation sets some classes appear in neither
    # y_true nor y_pred, and sklearn would otherwise reject target_names.
    return classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                 target_names=class_names, zero_division=0)


def confusion(y_true, y_pred, num_classes):
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def mcnemar(correct_a, correct_b):
    """Exact McNemar test between two arms scored on the same images.

    Arms evaluated on one dataset agree on most of it, so asking whether their
    independent confidence intervals overlap throws the pairing away along with
    most of the power. Only images where exactly one arm is right carry any
    information about which is better, and the exact binomial over those needs no
    large-sample approximation.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("a paired test needs one prediction per image from both arms")
    only_a = int((a & ~b).sum())
    only_b = int((b & ~a).sum())
    discordant = only_a + only_b
    return {
        "n": int(a.size),
        "accuracy_a": round(100.0 * float(a.mean()), 2),
        "accuracy_b": round(100.0 * float(b.mean()), 2),
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "discordant": discordant,
        "net_gain_b": only_b - only_a,
        "p_value": float(binomtest(only_b, discordant, 0.5).pvalue) if discordant else 1.0,
    }
