"""Disease severity estimation from image features.

Leaf disease severity is approximated as the fraction of leaf area showing
lesion symptoms. The leaf is segmented from the background by saturation, and
lesions are the leaf pixels falling outside the healthy-green hue band. The
resulting ratio is bucketed into ordinal severity levels.
"""

import cv2

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


def lesion_ratio(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
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


def estimate_severity(image_bgr, thresholds, levels):
    ratio = lesion_ratio(image_bgr)
    return severity_level(ratio, thresholds, levels), ratio
