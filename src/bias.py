"""Dataset-bias diagnostics for PlantVillage.

Two probes quantify how much of the headline accuracy comes from capture bias
rather than leaf pathology:

* ``border_features`` — samples a handful of background pixels from the image
  border. A classifier trained on these alone should be near chance (1/38) if
  the background carries no label information.
* ``segmented_path``  — maps a colour image to its background-removed twin in
  the ``segmented`` variant, so trained models can be re-scored without the
  background.
"""

import os

BORDER_POSITIONS = [(0, 0), (16, 0), (31, 0), (0, 16), (31, 16), (0, 31), (16, 31), (31, 31)]


def border_features(path, size=32, positions=BORDER_POSITIONS):
    """Return the RGB values of ``positions`` on a ``size``x``size`` thumbnail."""
    from PIL import Image

    image = Image.open(path)
    image.draft("RGB", (size, size))
    pixels = image.convert("RGB").resize((size, size)).load()
    return [channel for xy in positions for channel in pixels[xy]]


def name_key(filename):
    """Key that matches a colour file to its segmented twin.

    Colour and segmented filenames disagree: some classes carry a UUID prefix
    only in one variant, and segmented files end in ``_final_masked``.
    """
    stem = os.path.splitext(filename)[0]
    if "___" in stem:
        stem = stem.split("___", 1)[1]
    if stem.endswith("_final_masked"):
        stem = stem[: -len("_final_masked")]
    return stem


def segmented_index(segmented_root, class_name):
    """Map ``name_key`` -> path for one class of the segmented variant."""
    class_dir = os.path.join(segmented_root, class_name)
    return {name_key(f): os.path.join(class_dir, f) for f in os.listdir(class_dir)}
