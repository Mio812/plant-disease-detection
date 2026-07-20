# Automatic Plant Disease Detection Using Computer Vision

CNN-based classification of plant leaf diseases on the **PlantVillage** dataset —
three CNN baselines combined into a **soft-voting ensemble** — extended with an
image-feature-based **disease severity** estimate.

> COMP9444 25T1 — Project 090. The graded notebook is
> [`notebooks/plant_disease_detection.ipynb`](notebooks/plant_disease_detection.ipynb);
> the reusable code lives in [`src/`](src).

## Contents

- [Overview](#overview)
- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Setup with uv on Windows](#setup-with-uv-on-windows)
- [Usage](#usage)
- [Configuration](#configuration)
- [Methodology](#methodology)
- [Results](#results)
- [References](#references)

## Overview

Plant diseases threaten agricultural productivity and food security. Manual
inspection is slow and subjective, so this project trains a convolutional neural
network to classify leaf images into healthy and diseased crop/condition classes,
and estimates how advanced a disease is from the lesion area on the leaf.

The pipeline is modular and config-driven: data loading, models, training,
evaluation, and severity estimation are independent modules orchestrated by the
scripts and the notebook.

## Project structure

```
plant-disease-detection/
├── README.md
├── pyproject.toml              # uv project + dependencies (canonical)
├── uv.lock                     # pinned, reproducible resolution
├── requirements.txt            # pip fallback
├── .python-version             # interpreter pin for uv (3.12)
├── .vscode/                    # shared editor / debug configuration
├── configs/
│   └── default.yaml            # all hyper-parameters and paths
├── src/
│   ├── config.py               # typed YAML config loader
│   ├── data.py                 # transforms, splits, dataloaders
│   ├── models.py               # custom CNN + transfer-learning backbones
│   ├── engine.py               # train / evaluate loops, early stopping
│   ├── metrics.py              # accuracy, P/R/F1, confusion matrix
│   ├── severity.py             # lesion-area severity estimation
│   ├── ensemble.py             # soft-voting ensemble of the baselines
│   ├── bias.py                 # background-bias probes, segmented-mask matching
│   ├── crossdata.py            # PlantDoc -> PlantVillage class mapping
│   ├── visualize.py            # EDA and results plots
│   └── utils.py                # seed, device, checkpoints
├── scripts/
│   ├── download_data.py        # fetch PlantVillage into data/
│   ├── download_plantdoc.py    # fetch the PlantDoc field-image test split
│   ├── train.py                # train from a config
│   ├── train_robust.py         # background randomisation / segmented training
│   ├── evaluate.py             # evaluate a checkpoint on the test split
│   ├── predict.py              # classify + estimate severity for one image
│   ├── ensemble.py             # evaluate the soft-voting ensemble
│   ├── run_all.py              # resumable driver for the whole pipeline
│   ├── bias_probe.py           # predict the class from background pixels alone
│   ├── eval_segmented.py       # re-score models with the background removed
│   ├── eval_plantdoc.py        # zero-shot evaluation on field images
│   ├── severity_probe.py       # severity validation without manual labels
│   ├── severity_sample.py      # sample leaves for manual severity grading
│   ├── severity_validate.py    # score severity against manual grades
│   └── gradcam.py              # Grad-CAM plus leaf-attention audit
├── notebooks/
│   └── plant_disease_detection.ipynb
├── report/                     # summary report (.docx) and slides (.pptx)
└── outputs/                    # checkpoints, metrics, plots (created at runtime)
```

## Dataset

[PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) — ~54,000 leaf
images across 14 crop species and 38 crop/condition classes (26 diseases plus
healthy leaves), in colour, grayscale, and segmented variants. We train on the
colour images.

Folders are named `Crop___Condition` (e.g. `Tomato___Late_blight`) and are loaded
with `torchvision.datasets.ImageFolder`. After download, the colour images sit at
`data/PlantVillage/raw/color`, which is the default `data.root`.

## Setup with uv on Windows

The project is managed with [uv](https://docs.astral.sh/uv/). These steps target
**native Windows (PowerShell) + VSCode**; the same `uv` commands work on Linux and
macOS.

### 1. Install uv and Git

```powershell
# Install uv (skip if `uv --version` already works)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Git is required by `scripts/download_data.py`; install it from <https://git-scm.com>
if `git --version` fails.

### 2. Create the environment

From the project root, sync the extra that matches your hardware (exactly one):

```powershell
uv sync --extra cu130   # NVIDIA GPU (CUDA 13.0 wheels; supports RTX 50-series)
uv sync --extra cpu     # CPU only
```

`uv sync` creates `.venv\` and installs the locked dependencies plus the `dev`
group (Jupyter, Ruff). Verify the GPU build:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. Open in VSCode

Open the folder and install the recommended extensions when prompted. Run
**Python: Select Interpreter** and choose `.venv\Scripts\python.exe`. For the
notebook, select the same `.venv` kernel.

## Usage

Run everything from the project root via `uv run` (no manual activation needed):

```powershell
# 1. Download the dataset (once)
uv run python -m scripts.download_data

# 2. Train (reads configs/default.yaml)
uv run python -m scripts.train --config configs/default.yaml

# 3. Evaluate a checkpoint on the test split (saves a confusion matrix)
uv run python -m scripts.evaluate --config configs/default.yaml --checkpoint outputs/resnet18_best.pth

# 4. Classify a single leaf and estimate its severity
uv run python -m scripts.predict --config configs/default.yaml --checkpoint outputs/resnet18_best.pth --classes outputs/resnet18_history.json --image path/to/leaf.jpg

# 5. Evaluate the soft-voting ensemble of the three baselines
uv run python -m scripts.ensemble --config configs/default.yaml

# 6. Generalisation analysis (see "Reality check" below)
uv run python -m scripts.bias_probe          # class from background pixels only
uv run python -m scripts.eval_segmented      # background removed
uv run python -m scripts.download_plantdoc   # fetch field images (once)
uv run python -m scripts.eval_plantdoc       # zero-shot on field images

# 6b. Or run the whole generalisation pipeline in one resumable command
uv run python -m scripts.run_all --dry-run   # show the plan
uv run python -m scripts.run_all --quick     # 2-epoch smoke test (isolated in outputs_quick/)
uv run python -m scripts.run_all             # full run; re-run to resume after an interruption

# 6c. Explainability and label-free severity validation
uv run python -m scripts.gradcam --model resnet18 --n 60
uv run python -m scripts.severity_probe --n 150

# 7. Severity: sample for grading, then validate against manual grades
uv run python -m scripts.severity_sample --n 150
uv run python -m scripts.severity_validate --csv outputs/severity_annotations.csv
```

Open the notebook for the full analysis:

```powershell
uv run jupyter notebook notebooks/plant_disease_detection.ipynb
```

> **Windows note:** the CLI scripts use `num_workers > 0` from the config (the
> `__main__` guard makes the spawn start method safe). The notebook sets
> `num_workers = 0` on Windows to avoid the DataLoader/Jupyter spawn conflict.

> **Without uv (pip):** `py -m venv .venv` then `.venv\Scripts\pip install -r
> requirements.txt` also works; `pyproject.toml` + `uv.lock` is the canonical,
> reproducible source.

## Configuration

Every run is described by a YAML file; no hyper-parameters are hard-coded. Key
fields in [`configs/default.yaml`](configs/default.yaml):

| Group      | Field                | Description                                   |
| ---------- | -------------------- | --------------------------------------------- |
| `data`     | `root`               | Path to the `ImageFolder` directory           |
|            | `image_size`         | Square input resolution                       |
|            | `val_split` / `test_split` | Fractions held out for validation / test |
|            | `batch_size`         | Mini-batch size                               |
| `model`    | `name`               | `custom_cnn`, `resnet18`, or `mobilenet_v2`   |
|            | `pretrained`         | Use ImageNet weights for the backbone         |
| `train`    | `epochs`, `lr`, `weight_decay` | Optimisation schedule               |
|            | `label_smoothing`    | Cross-entropy label smoothing                 |
|            | `early_stop_patience`| Epochs without val-accuracy gain before stop  |
| `severity` | `thresholds` / `levels` | Lesion-ratio cut-offs and level names      |

## Methodology

**Classification.** Three baselines spanning two families are compared, then combined:

- **Custom CNN** — a four-block Conv–BatchNorm–ReLU–MaxPool network with global
  average pooling, trained from scratch (our baseline).
- **Transfer learning** — ImageNet-pretrained ResNet-18 / MobileNet-V2 with the
  final layer replaced for the PlantVillage classes. The backbone weights are the
  pre-existing source; the classifier head and the training pipeline are our work.
- **Soft-voting ensemble (proposed)** — a weighted average of the three baselines'
  softmax probabilities, weights tuned on the validation split (0.20 / 0.50 / 0.30).
  It needs no extra training and is our most accurate model.

Training uses AdamW with a cosine-annealed learning rate, cross-entropy with label
smoothing, on-the-fly augmentation (random resized crop, flip, rotation, colour
jitter), and early stopping on validation accuracy. Data is split 70/15/15 with a
fixed seed, and we report macro-averaged metrics to stay robust to class imbalance.

**Severity estimation.** PlantVillage has no severity labels, so severity is
derived from image features. The leaf is segmented from the background by
saturation, lesion pixels are the leaf pixels outside the healthy-green hue band,
and the lesion-area fraction is bucketed into ordinal levels (`healthy`, `mild`,
`moderate`, `severe`). It is applied only to leaves classified as diseased.

## Results

Test-set metrics (held-out 15%, 8,145 images; fixed seed → identical split for
every model):

| Model         | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
| ------------- | -------- | ----------------- | -------------- | ---------- |
| custom_cnn    | 0.9915   | 0.9901            | 0.9882         | 0.9891     |
| resnet18      | 0.9968   | 0.9958            | 0.9948         | 0.9951     |
| mobilenet_v2  | 0.9898   | 0.9867            | 0.9876         | 0.9870     |
| **ensemble**  | **0.9984** | **0.9975**      | **0.9978**     | **0.9976** |

**The ensemble wins.** All three backbones exceed 98.9% test accuracy; the from-scratch
custom CNN (99.15%) trails pretrained ResNet-18 by only ~0.5% and edges out
MobileNet-V2 — a small gap that reflects how separable PlantVillage is under
controlled conditions. ResNet-18 misclassifies only 26 / 8,145 test images; 12 of
those confuse Corn Cercospora/Gray leaf spot with Northern Leaf Blight (a known
visual look-alike), and no error crosses the healthy/diseased boundary. Averaging the
three baselines by a validation-tuned weighted vote lifts test accuracy to **99.84%** and cuts errors to
**13 / 8,145** — the best result, again with no healthy/diseased crossing; ResNet-18
(99.68%) remains the strongest  single model. Per-model training history, classification
report, and confusion matrix are in `outputs/`.

Each non-default row is reproduced with (same config, model overridden):

```powershell
uv run python -m scripts.train --model custom_cnn
uv run python -m scripts.train --model mobilenet_v2
```

## Reality check: is 99.8% real?

High PlantVillage accuracy is largely an artefact of the benchmark rather than
evidence of disease understanding. Four experiments quantify this.

**1. The background alone predicts the label.** A logistic regression trained on
**8 border pixels** — no leaf at all — reaches **31.5%** accuracy over 38 classes
(chance = 2.6%). The dataset carries capture bias correlated with the labels.

**2. Removing the background collapses accuracy.** Re-scoring the same models on
the same leaves using PlantVillage's `segmented` variant:

| Model         | Original | Background removed | Change |
| ------------- | -------- | ------------------ | ------ |
| custom_cnn    | 99.15%   | 33.90%             | −65.3  |
| resnet18      | 99.68%   | 56.75%             | −42.9  |
| mobilenet_v2  | 98.98%   | 63.95%             | −35.0  |
| **ensemble**  | 99.84%   | **57.60%**         | −42.2  |

The from-scratch CNN degrades most and the ImageNet-pretrained backbones least,
suggesting pretrained features attend more to the leaf and less to the backdrop.

**3. On real field photographs the models fail.** Zero-shot evaluation on the
[PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) test split (236
in-the-wild images, 27 classes mapped to PlantVillage):

| Model         | PlantVillage | PlantDoc (field) | Change |
| ------------- | ------------ | ---------------- | ------ |
| custom_cnn    | 99.15%       | 13.14%           | −86.0  |
| resnet18      | 99.68%       | 16.10%           | −83.6  |
| **ensemble**  | 99.84%       | **17.37%**       | −82.5  |

Coarse signal survives — healthy-vs-diseased binary accuracy is still 80.9% —
but fine-grained disease identification does not transfer. Test-time fixes do
not rescue it: hflip TTA adds +0.4 points and AdaBN *costs* 2.1, so the failure
is a learned shortcut rather than a statistics shift.

**4. The model only partly looks at the leaf.** Grad-CAM mass falling inside the
official leaf mask is **61.3%**, against a **47.5%** leaf-area baseline — a lift
of only +13.8 points, so well over a third of the evidence is background.

**Severity.** Two checks, neither needing manual labels. The HSV leaf segmentation
scored against the official masks gives Dice **0.79** mean / **0.88** median over
300 images (9.3% below 0.5). The lesion ratio separates healthy from diseased
leaves with **ROC-AUC 0.888** using the official masks versus **0.774** with Otsu
segmentation, which is why the official masks are preferred; mean lesion ratio is
0.035 on healthy leaves and 0.227 on diseased ones. `severity_sample.py` /
`severity_validate.py` additionally score the ordinal grade against manual
annotations (Spearman, MAE, quadratic κ).

**Task 1 as the brief words it.** For "healthy vs diseased", the ensemble and
ResNet-18 both reach **100.00%** on the held-out test split — none of their
errors cross the healthy/diseased boundary.

## References

1. Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on
   plant health.* arXiv:1511.08060.
2. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using deep learning for
   image-based plant disease detection.* Frontiers in Plant Science, 7, 1419.
3. Ferentinos, K. P. (2018). *Deep learning models for plant disease detection and
   diagnosis.* Computers and Electronics in Agriculture, 145, 311–318.
4. Arsenovic, M., et al. (2019). *Solving current limitations of deep learning based
   approaches for plant disease detection.* Symmetry, 11(7), 939.
5. Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020).
   *PlantDoc: A dataset for visual plant disease detection.* In Proc. 7th ACM IKDD
   CoDS and 25th COMAD, 249–253.
6. Noyan, M. A. (2022). *Uncovering bias in the PlantVillage dataset.*
   arXiv:2206.04374.
7. Natarajan, S., Chakrabarti, P., & Margala, M. (2024). *Robust diagnosis and meta
   visualizations of plant diseases through deep neural architecture with
   explainable AI.* Scientific Reports, 14, 13695.
