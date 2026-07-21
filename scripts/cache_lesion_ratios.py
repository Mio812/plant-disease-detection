"""Cache the official-mask lesion ratio for every image, once.

E25's regression target, computed for the whole corpus. Ratios come from the
official masks, never Otsu -- reading each segmented twin costs two decodes, hence
the cache.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from torchvision.datasets import ImageFolder

from src.audit.severity import leaf_mask_from_segmented, lesion_ratio
from src.config import Config
from src.data.variants import name_key, variant_index, variant_root


def parse_args():
    parser = argparse.ArgumentParser(description="Cache official-mask lesion ratios.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=None, help="stop early, for smoke tests")
    parser.add_argument("--out", default="outputs/lesion_ratios.json")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    base = ImageFolder(cfg.data.root)
    seg_root = variant_root(cfg.data.root, "segmented")

    index, ratios, missing, unreadable = {}, {}, 0, 0
    samples = base.samples[:args.limit] if args.limit else base.samples

    for n, (path, label) in enumerate(samples, 1):
        class_name = base.classes[label]
        if class_name not in index:
            index[class_name] = variant_index(seg_root, class_name)
        twin = index[class_name].get(name_key(Path(path).name))
        if twin is None:
            missing += 1
            continue
        image, seg = cv2.imread(path), cv2.imread(twin)
        if image is None or seg is None:
            unreadable += 1
            continue
        mask = leaf_mask_from_segmented(cv2.resize(seg, image.shape[1::-1]))
        if not mask.any():
            unreadable += 1
            continue
        ratios[Path(path).name] = round(float(lesion_ratio(image, mask)), 6)
        if n % 5000 == 0:
            print(f"  {n:,}/{len(samples):,}  cached {len(ratios):,}", flush=True)

    values = np.array(list(ratios.values()), dtype=np.float32)
    summary = {
        "n_images": len(samples),
        "n_cached": len(ratios),
        "n_missing_twin": missing,
        "n_unreadable": unreadable,
        "mean": round(float(values.mean()), 6),
        "median": round(float(np.median(values)), 6),
        "p05": round(float(np.percentile(values, 5)), 6),
        "p95": round(float(np.percentile(values, 95)), 6),
        "ratios": ratios,
    }
    Path(args.out).write_text(json.dumps(summary), encoding="utf-8")
    print(f"\ncached {len(ratios):,}/{len(samples):,}  "
          f"(missing twin {missing}, unreadable {unreadable})")
    print(f"ratio  mean {summary['mean']:.4f}  median {summary['median']:.4f}  "
          f"p05 {summary['p05']:.4f}  p95 {summary['p95']:.4f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
