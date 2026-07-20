"""Cross-dataset evaluation on PlantDoc field images.

PlantDoc contains in-the-wild photographs (cluttered backgrounds, multiple
leaves) whose class names differ from PlantVillage's. The mapping below sends
each PlantDoc test class to its PlantVillage equivalent so that a model trained
on PlantVillage can be scored zero-shot.
"""

import math
import os

PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
}


def plantdoc_items(root, class_to_idx):
    """List ``(image_path, plantvillage_label)`` for a PlantDoc split directory."""
    items = []
    for folder in sorted(os.listdir(root)):
        target = PLANTDOC_TO_PLANTVILLAGE.get(folder)
        if target is None:
            continue
        label = class_to_idx[target]
        folder_path = os.path.join(root, folder)
        items.extend((os.path.join(folder_path, f), label) for f in sorted(os.listdir(folder_path)))
    return items


def wilson_interval(correct, total, z=1.96):
    """95% Wilson confidence interval for an accuracy - small test sets need it."""
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
