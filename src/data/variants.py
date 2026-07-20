"""Mapping one PlantVillage variant (colour / grayscale / segmented) onto another.

Filenames disagree between variants -- some classes carry a UUID prefix in only
one variant, and segmented files end in ``_final_masked`` -- so images are
matched on the original-name key rather than on the path. The split always comes
from the colour ImageFolder, so every variant sees the same physical leaves.
"""

import os
from pathlib import Path


def name_key(filename):
    """Key that identifies the same photograph across dataset variants."""
    stem = os.path.splitext(filename)[0]
    if "___" in stem:
        stem = stem.split("___", 1)[1]
    if stem.endswith("_final_masked"):
        stem = stem[: -len("_final_masked")]
    return stem


def variant_index(variant_dir, class_name):
    """Map ``name_key`` -> path for one class of a variant directory."""
    class_dir = os.path.join(variant_dir, class_name)
    return {name_key(f): os.path.join(class_dir, f) for f in os.listdir(class_dir)}


def variant_root(colour_root, variant):
    """Path of a sibling PlantVillage variant directory."""
    return str(Path(colour_root).parent / variant)


def variant_samples(base, indices, variant, variant_dir):
    """Map dataset indices onto ``variant``, returning ``(path, label)`` pairs."""
    if variant == "color":
        return [base.samples[i] for i in indices]
    index, items = {}, []
    for i in indices:
        path, label = base.samples[i]
        class_name = base.classes[label]
        if class_name not in index:
            index[class_name] = variant_index(variant_dir, class_name)
        twin = index[class_name].get(name_key(Path(path).name))
        if twin is not None:
            items.append((twin, label))
    return items
