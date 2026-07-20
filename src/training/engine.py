"""Training and evaluation loops with early stopping."""

import copy

import torch
from tqdm import tqdm

from ..evaluation.metrics import compute_metrics
from ..utils import AverageMeter, save_checkpoint


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    correct, total = 0, 0
    for images, targets in tqdm(loader, leave=False, desc="train"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), images.size(0))
        correct += (outputs.argmax(1) == targets).sum().item()
        total += images.size(0)
    return loss_meter.avg, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    y_true, y_pred = [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss_meter.update(loss.item(), images.size(0))
        y_pred.extend(outputs.argmax(1).cpu().tolist())
        y_true.extend(targets.cpu().tolist())
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = loss_meter.avg
    return metrics, y_true, y_pred


def fit(model, loaders, criterion, optimizer, scheduler, device, epochs, patience, ckpt_path):
    history = []
    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_metrics, _, _ = evaluate(model, loaders["val"], criterion, device)
        if scheduler is not None:
            scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )
        print(
            f"[{epoch:02d}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint({"model": best_state, "val_acc": best_acc, "epoch": epoch}, ckpt_path)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch} (best val_acc={best_acc:.4f})")
                break

    model.load_state_dict(best_state)
    return history
