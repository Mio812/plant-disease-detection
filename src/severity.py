"""Disease severity estimation from image features.

Severity is approximated as the fraction of leaf area showing lesion symptoms.
The leaf is segmented from the background (by saturation, or from PlantVillage's
official ``segmented`` mask when available) and lesions are the leaf pixels
falling outside the healthy-green hue band. The ratio is bucketed into ordinal
severity levels.
"""

import cv2
import numpy as np

HEALTHY_HUE_LOW = 35
HEALTHY_HUE_HIGH = 85


def _leaf_mask(hsv):
    saturation = hsv[:, :, 1]
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask > 0


def _lesion_mask(hsv, leaf):
    hue = hsv[:, :, 0]
    healthy = (hue >= HEALTHY_HUE_LOW) & (hue <= HEALTHY_HUE_HIGH)
    return leaf & ~healthy


def leaf_mask_from_segmented(segmented_bgr, threshold=25):
    """Leaf mask read from PlantVillage's official background-removed image."""
    return segmented_bgr.sum(axis=2) > threshold


def dice(mask_a, mask_b):
    """Dice overlap between two boolean masks."""
    denominator = int(mask_a.sum()) + int(mask_b.sum())
    if denominator == 0:
        return 0.0
    return 2 * int(np.logical_and(mask_a, mask_b).sum()) / denominator


def lesion_ratio(image_bgr, leaf=None):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if leaf is None:
        leaf = _leaf_mask(hsv)
    leaf_area = int(leaf.sum())
    if leaf_area == 0:
        return 0.0
    return int(_lesion_mask(hsv, leaf).sum()) / leaf_area


def severity_level(ratio, thresholds, levels):
    for level, threshold in zip(levels[:-1], thresholds, strict=False):
        if ratio < threshold:
            return level
    return levels[-1]


def estimate_severity(image_bgr, thresholds, levels, leaf=None):
    ratio = lesion_ratio(image_bgr, leaf)
    return severity_level(ratio, thresholds, levels), ratio
