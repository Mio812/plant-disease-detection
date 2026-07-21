"""Diagnostic probes for the classifier and the severity estimator.

    --probe background         predict the class from background pixels alone
    --probe gradcam            share of explanation mass landing on the leaf
    --probe severity           label-free severity validation (Dice + ROC-AUC)
    --probe efficiency         parameters, checkpoint size, inference latency
    --probe adaptation         test-time fixes (TTA, AdaBN) on field images
    --probe severity-sample    emit a CSV of leaves for manual grading
    --probe severity-validate  score the ordinal grade against manual grades

Usage:
    python -m scripts.audit --probe background
    python -m scripts.audit --probe gradcam --model resnet18
"""

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from src.audit import border_features, grad_cam, leaf_attention, target_layer
from src.audit.severity import (
    _leaf_mask,
    dice,
    estimate_severity,
    leaf_mask_from_segmented,
    lesion_ratio,
    severity_level,
)
from src.config import Config
from src.data import build_transforms, name_key, splits_from_config, variant_index, variant_root
from src.evaluation import load_model
from src.utils import get_device, set_seed


def _base_and_segmented(cfg):
    base = ImageFolder(cfg.data.root)
    return base, variant_root(cfg.data.root, "segmented")


def probe_background(cfg, args):
    """A classifier trained on border pixels alone should sit near 1/38."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    base = ImageFolder(cfg.data.root)
    train_idx, _, test_idx = splits_from_config(cfg, base)
    rows, seen = [], {}
    for split, indices, cap in (("train", train_idx, args.per_class_train),
                                ("test", test_idx, args.per_class_test)):
        for i in indices:
            _, label = base.samples[i]
            key = (split, label)
            if seen.get(key, 0) >= cap:
                continue
            seen[key] = seen.get(key, 0) + 1
            rows.append((base.samples[i][0], label, split == "train"))

    X = np.array([border_features(p) for p, _, _ in rows], dtype=np.float32)
    y = np.array([lab for _, lab, _ in rows])
    is_train = np.array([tr for _, _, tr in rows])
    scaler = StandardScaler().fit(X[is_train])
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X[is_train]), y[is_train])
    accuracy = float(clf.score(scaler.transform(X[~is_train]), y[~is_train]))
    print(f"  border pixels used : {X.shape[1] // 3}")
    print(f"  train {int(is_train.sum())} / test {int((~is_train).sum())} over {len(base.classes)} classes")
    print(f"  TEST ACCURACY = {accuracy * 100:.1f}%   (chance {100 / len(base.classes):.1f}%)")
    return "bias_probe.json", {"n_pixels": X.shape[1] // 3, "test_accuracy": accuracy,
                               "chance": 1 / len(base.classes)}


def probe_gradcam(cfg, args):
    base, seg_root = _base_and_segmented(cfg)
    device = get_device()
    checkpoint = args.checkpoint or f"{cfg.output_dir}/{args.model}_best.pth"
    model = load_model(args.model, len(base.classes), checkpoint, device)
    transform = build_transforms(cfg.data.image_size, train=False)

    rng = random.Random(cfg.seed)
    diseased = [i for i, (_, l) in enumerate(base.samples)
                if "healthy" not in base.classes[l].lower()]
    chosen = rng.sample(diseased, min(args.n, len(diseased)))
    index, inside, area = {}, [], []
    size = (cfg.data.image_size, cfg.data.image_size)

    for start in range(0, len(chosen), 16):
        tensors, masks = [], []
        for i in chosen[start:start + 16]:
            path, label = base.samples[i]
            class_name = base.classes[label]
            if class_name not in index:
                index[class_name] = variant_index(seg_root, class_name)
            twin = index[class_name].get(name_key(Path(path).name))
            seg = cv2.imread(twin) if twin else None
            if seg is None:
                continue
            masks.append(cv2.resize(seg, size).sum(axis=2) > 25)
            tensors.append(transform(Image.open(path).convert("RGB")))
        if not tensors:
            continue
        cams, _ = grad_cam(model, target_layer(model, args.model), torch.stack(tensors), device)
        for cam, mask in zip(cams, masks):
            a, b = leaf_attention(cam, mask)
            inside.append(a)
            area.append(b)

    summary = {"model": args.model, "n": len(inside),
               "mean_attention_in_leaf": float(np.mean(inside)),
               "mean_leaf_area_fraction": float(np.mean(area)),
               "attention_lift_over_area": float(np.mean(inside) - np.mean(area))}
    print(f"  Grad-CAM mass inside the leaf : {summary['mean_attention_in_leaf'] * 100:.1f}%")
    print(f"  leaf share of image area      : {summary['mean_leaf_area_fraction'] * 100:.1f}%")
    print(f"  lift over the area baseline   : {summary['attention_lift_over_area'] * 100:+.1f} points")
    return "gradcam_audit.json", summary


def probe_severity(cfg, args):
    """Without manual labels: does the lesion ratio separate healthy from diseased?"""
    from sklearn.metrics import roc_auc_score

    base, seg_root = _base_and_segmented(cfg)
    healthy_idx, diseased_idx = [], []
    for i, (_, label) in enumerate(base.samples):
        (healthy_idx if "healthy" in base.classes[label].lower() else diseased_idx).append(i)
    rng = random.Random(cfg.seed)
    chosen = ([(i, 0) for i in rng.sample(healthy_idx, min(args.n, len(healthy_idx)))] +
              [(i, 1) for i in rng.sample(diseased_idx, min(args.n, len(diseased_idx)))])

    index, rows, dices = {}, [], []
    for i, is_diseased in chosen:
        path, label = base.samples[i]
        image = cv2.imread(path)
        if image is None:
            continue
        class_name = base.classes[label]
        if class_name not in index:
            index[class_name] = variant_index(seg_root, class_name)
        twin = index[class_name].get(name_key(Path(path).name))
        official = None
        if twin:
            seg = cv2.imread(twin)
            if seg is not None:
                mask = leaf_mask_from_segmented(cv2.resize(seg, image.shape[1::-1]))
                official = lesion_ratio(image, mask)
                dices.append(dice(_leaf_mask(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)), mask))
        rows.append((is_diseased, lesion_ratio(image), official))

    y = np.array([r[0] for r in rows])
    otsu = np.array([r[1] for r in rows])
    have = np.array([r[2] is not None for r in rows])
    official = np.array([r[2] if r[2] is not None else 0.0 for r in rows])
    levels, thresholds = list(cfg.severity.levels), list(cfg.severity.thresholds)
    grades = [severity_level(r, thresholds, levels) for r in official[have & (y == 1)]]

    summary = {"n": len(rows),
               "auc_otsu_mask": float(roc_auc_score(y, otsu)),
               "auc_official_mask": float(roc_auc_score(y[have], official[have])),
               "mean_ratio_healthy_official": float(official[have & (y == 0)].mean()),
               "mean_ratio_diseased_official": float(official[have & (y == 1)].mean()),
               "diseased_grade_distribution": {lv: grades.count(lv) for lv in levels},
               "leaf_segmentation_dice_mean": float(np.mean(dices)) if dices else None,
               "leaf_segmentation_dice_median": float(np.median(dices)) if dices else None,
               "leaf_segmentation_dice_below_0.5": float(np.mean(np.array(dices) < 0.5)) if dices else None}
    print(f"  ROC-AUC healthy vs diseased, Otsu mask     : {summary['auc_otsu_mask']:.3f}")
    print(f"  ROC-AUC healthy vs diseased, official mask : {summary['auc_official_mask']:.3f}")
    print(f"  mean lesion ratio  healthy {summary['mean_ratio_healthy_official']:.3f} "
          f"| diseased {summary['mean_ratio_diseased_official']:.3f}")
    print(f"  diseased grades: {summary['diseased_grade_distribution']}")
    if dices:
        print(f"  leaf-segmentation Dice vs official masks: mean "
              f"{summary['leaf_segmentation_dice_mean']:.3f} / median "
              f"{summary['leaf_segmentation_dice_median']:.3f}")
    return "severity_probe.json", summary


def probe_adaptation(cfg, args):
    """E7 - can inference-time fixes repair the field gap? (Expected: no.)"""
    from torch.utils.data import DataLoader

    from src.data import ItemDataset, plantdoc_items
    from src.evaluation import enable_batchnorm_adaptation, load_model, predict_loader
    from src.models import combine

    base = ImageFolder(cfg.data.root)
    classes = base.classes
    items = plantdoc_items(args.plantdoc, {c: i for i, c in enumerate(classes)})
    loader = DataLoader(ItemDataset(items, build_transforms(cfg.data.image_size, train=False)),
                        batch_size=cfg.data.batch_size, num_workers=0)
    device = get_device()
    weights = args.weights
    members = ["custom_cnn", "resnet18", "mobilenet_v2"]

    y = np.array([lab for _, lab in items])
    present = np.zeros(len(classes), bool)
    present[sorted(set(y.tolist()))] = True

    # load each member once; AdaBN only flips BatchNorm mode, so no reloading
    loaded = [load_model(n, len(classes), f"{cfg.output_dir}/{n}_best.pth", device)
              for n in members]

    def score(adabn, tta):
        probs = []
        for model in loaded:
            model.eval()
            if adabn:
                enable_batchnorm_adaptation(model)
            member, _ = predict_loader(model, loader, device, tta=tta)
            probs.append(member)
        ens = combine(probs, weights).numpy()
        closed = ens.copy()
        closed[:, ~present] = 0.0
        return (float((ens.argmax(1) == y).mean() * 100),
                float((closed.argmax(1) == y).mean() * 100))

    summary = {}
    for label, adabn, tta in [("baseline", False, False), ("+TTA", False, True),
                              ("+AdaBN", True, False), ("+AdaBN +TTA", True, True)]:
        open_acc, closed_acc = score(adabn, tta)
        summary[label] = {"open_38way": round(open_acc, 2), "closed_27way": round(closed_acc, 2)}
        print(f"  {label:14s} 38-way {open_acc:5.2f}%   closed-set {closed_acc:5.2f}%")
    return "adaptation_probe.json", summary


def probe_efficiency(cfg, args):
    """Deployment profile: size and latency, for the edge-deployment claim."""
    import time

    base = ImageFolder(cfg.data.root)
    device = get_device()
    size = args.image_size or cfg.data.image_size
    rows = {}
    for name in ["custom_cnn", "resnet18", "mobilenet_v2"]:
        checkpoint = Path(cfg.output_dir) / f"{name}_best.pth"
        model = load_model(name, len(base.classes), checkpoint, device)
        params = sum(p.numel() for p in model.parameters())
        x = torch.randn(1, 3, size, size, device=device)
        with torch.no_grad():
            for _ in range(5):
                model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.latency_runs):
                model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            latency = (time.perf_counter() - t0) / args.latency_runs * 1000
        rows[name] = {"parameters_M": round(params / 1e6, 2),
                      "checkpoint_MB": round(checkpoint.stat().st_size / 1e6, 1),
                      "latency_ms": round(latency, 2),
                      "throughput_img_s": round(1000 / latency, 1)}
        print(f"  {name:14s} {rows[name]['parameters_M']:6.2f}M params  "
              f"{rows[name]['checkpoint_MB']:6.1f} MB  "
              f"{rows[name]['latency_ms']:7.2f} ms/img  "
              f"{rows[name]['throughput_img_s']:7.1f} img/s")
    rows["device"] = str(device)
    rows["image_size"] = size
    return "efficiency_probe.json", rows


def probe_severity_sample(cfg, args):
    base = ImageFolder(cfg.data.root)
    diseased = [i for i, (_, l) in enumerate(base.samples)
                if "healthy" not in base.classes[l].lower()]
    chosen = random.Random(cfg.seed).sample(diseased, min(args.n, len(diseased)))
    out = Path(cfg.output_dir) / "severity_annotations.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "class", "lesion_ratio", "estimated_level", "manual_grade"])
        for i in chosen:
            path, label = base.samples[i]
            level, ratio = estimate_severity(cv2.imread(path), cfg.severity.thresholds,
                                             cfg.severity.levels)
            writer.writerow([path, base.classes[label], f"{ratio:.4f}", level, ""])
    print(f"  wrote {len(chosen)} rows to {out.as_posix()} - fill in `manual_grade` (0-3)")
    return None, None


def probe_severity_validate(cfg, args):
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score

    levels = list(cfg.severity.levels)
    ratios, predicted, manual = [], [], []
    with open(args.csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["manual_grade"].strip():
                continue
            ratios.append(float(row["lesion_ratio"]))
            predicted.append(levels.index(row["estimated_level"]))
            manual.append(int(row["manual_grade"]))
    if not manual:
        raise SystemExit("No graded rows - fill in the manual_grade column first.")
    predicted, manual = np.array(predicted), np.array(manual)
    rho, p = spearmanr(ratios, manual)
    summary = {"n_graded": len(manual), "spearman_rho": float(rho), "spearman_p": float(p),
               "mae_levels": float(np.abs(predicted - manual).mean()),
               "exact_agreement": float((predicted == manual).mean()),
               "quadratic_kappa": float(cohen_kappa_score(predicted, manual, weights="quadratic"))}
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return "severity_validation.json", summary


def probe_paired(cfg, args):
    """McNemar between two arms scored on the same images (E8, E15)."""
    from src.evaluation import mcnemar
    from src.models.hierarchical import crop_of

    if not args.arms or len(args.arms) != 2:
        raise SystemExit("--arms takes exactly two *_predictions.json files")
    a, b = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.arms]
    if a["true"] != b["true"]:
        raise SystemExit("the two arms were not scored on the same images in the same order")

    classes, truth = a["classes"], a["true"]
    crop = crop_of(classes)[0].tolist()
    healthy = [1 if "healthy" in c.lower() else 0 for c in classes]
    views = {"all_38": lambda p, t: p == t,
             "crop": lambda p, t: crop[p] == crop[t],
             "binary": lambda p, t: healthy[p] == healthy[t]}

    def short(path):
        return Path(path).stem.replace("eval_arm_", "").replace("_predictions", "")

    names = [short(p) for p in args.arms]
    summary = {"arm_a": names[0], "arm_b": names[1], "n": len(truth)}
    print(f"  {names[0]} (a)  vs  {names[1]} (b)")
    for level, hit in views.items():
        result = mcnemar([hit(p, t) for p, t in zip(a["pred"], truth)],
                         [hit(p, t) for p, t in zip(b["pred"], truth)])
        summary[level] = result
        print(f"  {level:8s} {result['accuracy_a']:6.2f} vs {result['accuracy_b']:6.2f}   "
              f"net {result['net_gain_b']:+5d} of {result['discordant']:5d} discordant   "
              f"p = {result['p_value']:.4g}")
    return f"paired_{names[0]}_vs_{names[1]}.json", summary


PROBES = {"background": probe_background, "gradcam": probe_gradcam, "severity": probe_severity,
          "adaptation": probe_adaptation, "efficiency": probe_efficiency, "paired": probe_paired,
          "severity-sample": probe_severity_sample, "severity-validate": probe_severity_validate}


def main():
    parser = argparse.ArgumentParser(description="Audit probes for the classifier and severity.")
    parser.add_argument("--probe", choices=sorted(PROBES), required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--per-class-train", type=int, default=100)
    parser.add_argument("--per-class-test", type=int, default=50)
    parser.add_argument("--csv", default="outputs/severity_annotations.csv")
    parser.add_argument("--arms", nargs="+", default=None,
                        help="two *_predictions.json files, for --probe paired")
    parser.add_argument("--plantdoc", default="data/PlantDoc/test")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.2, 0.5, 0.3])
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--latency-runs", type=int, default=50)
    parser.add_argument("--out", default=None,
                        help="override the output filename (use for smoke tests)")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    print(f"probe: {args.probe}")
    filename, summary = PROBES[args.probe](cfg, args)
    if filename:
        out = Path(args.out) if args.out else Path(cfg.output_dir) / filename
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  saved {out.as_posix()}")


if __name__ == "__main__":
    main()
