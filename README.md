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
│   ├── severity_sample.py      # sample leaves for manual severity grading
│   └── severity_validate.py    # score severity against manual grades
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
(99.68%) remains the strongest 