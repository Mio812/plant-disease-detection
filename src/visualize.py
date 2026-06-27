"""Plotting helpers for exploratory analysis and results."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_class_distribution(counts, top=None):
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    if top:
        items = items[:top]
    labels, values = zip(*items, strict=False)
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.3)))
    ax.barh(labels, values, color="seagreen")
    ax.invert_yaxis()
    ax.set_xlabel("Number of images")
    ax.set_title("Class distribution")
    fig.tight_layout()
    return fig


def plot_history(history):
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax_loss.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax_loss.legend()
    ax_acc.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax_acc.plot(epochs, [h["val_accuracy"] for h in history], label="val")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    ax_acc.legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names, normalize=True):
    matrix = cm.astype(float)
    if normalize:
        matrix /= matrix.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matrix, xticklabels=class_names, yticklabels=class_names, cmap="viridis", ax=ax, cbar=True
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig


def show_samples(dataset, class_names, n=8):
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for ax, idx in zip(np.ravel(axes), range(n), strict=False):
        image, label = dataset[idx]
        if hasattr(image, "permute"):
            image = image.permute(1, 2, 0).numpy()
            image = (image - image.min()) / (np.ptp(image) + 1e-8)
        ax.imshow(image)
        ax.set_title(class_names[label], fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    return fig
