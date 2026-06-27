"""Classification metrics and reports."""

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
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)


def confusion(y_true, y_pred, num_classes):
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
