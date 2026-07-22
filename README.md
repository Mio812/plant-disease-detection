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
├── run.ps1                     # one-command driver (Windows)
├── pyproject.toml              # uv project + dependencies (canonical)
├── configs/
│   ├── default.yaml            # hyper-parameters and paths
│   └── experiments.yaml        # the declarative experiment matrix
├── docs/
│   └── EXPERIMENTS.md          # research question, hypotheses, experiment plan
├── src/
│   ├── config.py               # typed YAML config loader
│   ├── utils.py                # seed, device, checkpoints
│   ├── data/                   # splits (single source of truth), variants,
│   │                           #   PlantVillage loaders, PlantDoc mapping
│   ├── models/                 # architectures + soft-voting ensemble
│   ├── training/               # optimisation loops, augmentation, background randomisation
│   ├── evaluation/             # metrics, one inference path, plots
│   └── audit/                  # background probe, Grad-CAM, severity
├── scripts/
│   ├── prepare_data.py         # download PlantVillage and PlantDoc
│   ├── train.py                # every training variant behind one entry point
│   ├── evaluate.py             # --on plantvillage | segmented | grayscale | plantdoc
│   ├── ensemble.py             # validation-tuned soft voting
│   ├── audit.py                # --probe background | gradcam | severity | ...
│   ├── finetune.py             # supervised adaptation on PlantDoc
│   ├── annotate.py             # package/merge the severity annotation task
│   ├── predict.py              # classify + grade a single leaf
│   └── run_all.py              # executes configs/experiments.yaml
├── notebooks/
│   └── plant_disease_detection.ipynb
├── report/                     # summary report (.docx) and slides (.pptx)
└── outputs/                    # checkpoints, metrics, figures (created at runtime)
```

## Dataset

[PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) — ~54,000 leaf
images across 14 crop species and 38 crop/condition classes (26 diseases plus
healthy leaves), in colour, grayscale, and segmented variants. We train on the
colour images.

Folders are named `Crop___Condition` (e.g. `Tomato___Late_blight`) and are loaded
with `torchvision.datasets.ImageFolder`. After download, the colour images sit at
`data/PlantVillage/raw/color`, which is the default `data.root`.

[PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) — 2,525 in-the-wild
field photographs used **only** for external validation; never trained on.

## Reproducibility and data provenance

Every number in the report, notebook, and README is traceable, not asserted:

- **Datasets are public and cited**, not ours: PlantVillage (Hughes & Salathé /
  Mohanty et al.) and PlantDoc (Singh et al.). `scripts.prepare_data` downloads both;
  they are gitignored only because of size, and their sources are linked above.
- **Every result is reproducible** from a single fixed seed and one shared 70/15/15
  split. `uv run python -m scripts.run_all` regenerates the full experiment matrix.
- **The raw result artefacts are committed** under `outputs/*.json` (50 files — every
  evaluation, probe, and audit that backs a reported number) and `outputs/*.txt`
  (per-class classification reports). Trained checkpoints (~696 MB) are omitted for
  size but regenerate deterministically from the seed.
- **The notebook carries embedded outputs**, so all results are visible without
  re-running anything.
- **The 150 human severity grades** (three annotators) — the only primary data we
  produced — are committed under `outputs/annotation/` alongside the grading tool
  and instructions, so `--probe severity-validate` is fully reproducible.

## Setup with uv on Windows

The project is managed with [uv](https://docs.astral.sh/uv/). These steps target
**native Windows (PowerShell) + VSCode**; the same `uv` commands work on Linux and
macOS.

### 1. Install uv and Git

```powershell
# Install uv (skip if `uv --version` already works)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Git is required by `scripts/prepare_data.py`; install it from <https://git-scm.com>
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

Run everything from the project root via `uv run` (no manual activation needed).
The whole experiment matrix is one resumable command:

```powershell
.\run.ps1                                    # environment check, then the full pipeline
uv run python -m scripts.run_all --dry-run   # show the plan without running anything
uv run python -m scripts.run_all             # re-run to resume after an interruption
```

The individual entry points, if you want to run a single step:

```powershell
# Data
uv run python -m scripts.prepare_data --dataset both

# Training: baselines, background randomisation, and the variant ablations
uv run python -m scripts.train --model resnet18
uv run python -m scripts.train --model resnet18 --augment strong --p-random 0.7 --image-size 224
uv run python -m scripts.train --model resnet18 --variant segmented --augment strong --image-size 224

# Evaluation on any dataset
uv run python -m scripts.evaluate --on plantvillage --weights 0.2 0.5 0.3 --save-reports
uv run python -m scripts.evaluate --on segmented --weights 0.2 0.5 0.3
uv run python -m scripts.evaluate --on plantdoc --plantdoc data/PlantDoc/test

# Ensemble weights tuned on the validation split
uv run python -m scripts.ensemble

# Audit probes
uv run python -m scripts.audit --probe background     # class from border pixels only
uv run python -m scripts.audit --probe gradcam        # explanation mass on the leaf
uv run python -m scripts.audit --probe adaptation     # test-time fixes (TTA, AdaBN)
uv run python -m scripts.audit --probe severity       # label-free severity validation
uv run python -m scripts.audit --probe efficiency     # size and latency

# Supervised adaptation on field images
uv run python -m scripts.finetune --model resnet18 --checkpoint outputs/resnet18_best.pth --shots 20

# Severity annotation: package for the team, merge, then score
uv run python -m scripts.annotate --action package --annotators 4
uv run python -m scripts.annotate --action merge
uv run python -m scripts.audit --probe severity-validate

# Classify a single leaf
uv run python -m scripts.predict --checkpoint outputs/resnet18_best.pth --classes outputs/resnet18_history.json --image path/to/leaf.jpg
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

Test-set metrics on the **leaf-grouped** split (held-out 15%, 8,215 images; no
physical leaf shared with training — see *Test-set integrity* below):

| Model         | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
| ------------- | -------- | ----------------- | -------------- | ---------- |
| custom_cnn    | 0.9869   | 0.9843            | 0.9821         | 0.9831     |
| resnet18      | 0.9933   | 0.9919            | 0.9922         | 0.9919     |
| mobilenet_v2  | 0.9946   | 0.9903            | 0.9926         | 0.9914     |
| **ensemble**  | **0.9953** | **0.9946**      | **0.9939**     | **0.9942** |

**The ensemble wins.** All three backbones exceed 98.6% test accuracy on unseen
leaves. Averaging them by a validation-tuned weighted vote lifts accuracy to
**99.53%** over the 38 specific-disease classes and **99.96%** on the healthy-vs-diseased
task the brief names — with **39 / 8,215** errors, almost all within-crop disease
look-alikes (e.g. Tomato early vs late blight) and **no** healthy/diseased crossing.
Per-model history, classification report, and confusion matrix are in `outputs/`.

Each non-default row is reproduced with (same config, model overridden):

```powershell
uv run python -m scripts.train --model custom_cnn
uv run python -m scripts.train --model mobilenet_v2
```

## Reality check: is 99.5% real?

High PlantVillage accuracy is largely an artefact of the benchmark rather than
evidence of disease understanding. Several experiments quantify this.

**0. Test-set integrity.** PlantVillage photographs each leaf several times, and its
`leaf-map.json` records which images share a leaf. A naive random split leaks those
near-duplicates: 74.7% of a random test split had a same-leaf twin in training. We
split by *leaf* instead, so no leaf straddles the partition. This is the split used
everywhere here; adopting it lowered the ensemble from a leaked 99.84% to an honest
99.53%, and per disease it exposed Tomato early blight falling from a leaked 99% to
an honest 91% — the one diagnostic weakness the leak had hidden.

**1. The background alone predicts the label.** A logistic regression trained on
**8 border pixels** — no leaf at all — reaches **33.7%** accuracy over 38 classes
(chance = 2.6%). The dataset carries capture bias correlated with the labels.

**2. Removing the background collapses accuracy.** Re-scoring the same models on
the same leaves using PlantVillage's `segmented` variant:

| Model         | Original | Background removed | Change |
| ------------- | -------- | ------------------ | ------ |
| custom_cnn    | 98.69%   | 38.48%             | −60.2  |
| resnet18      | 99.33%   | 69.57%             | −29.8  |
| mobilenet_v2  | 99.46%   | 71.41%             | −28.1  |
| **ensemble**  | 99.53%   | **70.05%**         | −29.5  |

The from-scratch CNN degrades most and the ImageNet-pretrained backbones least,
suggesting pretrained features attend more to the leaf and less to the backdrop.

**3. On real field photographs the models fail.** Zero-shot evaluation on the
[PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) test split (236
in-the-wild images, 27 classes mapped to PlantVillage):

| Model         | PlantVillage | PlantDoc (field) | Change |
| ------------- | ------------ | ---------------- | ------ |
| custom_cnn    | 98.69%       | 13.56%           | −85.1  |
| resnet18      | 99.33%       | 14.41%           | −84.9  |
| **ensemble**  | 99.53%       | **16.10%**       | −83.4  |

Coarse signal survives — healthy-vs-diseased binary accuracy is still 80.5% —
but fine-grained disease identification does not transfer. Test-time fixes do
not rescue it, so the failure is a learned shortcut rather than a statistics
shift. The bottleneck is crop identification (39.6% in the field), not diagnosis.

**4. The model only partly looks at the leaf.** Grad-CAM mass falling inside the
official leaf mask is **65.6%**, against a **49.6%** leaf-area baseline — a lift
of only +16 points, so a third of the evidence is still background.

**Severity (Task 2).** The lesion-area ratio separates healthy from diseased leaves
with **ROC-AUC 0.868** using the official masks versus **0.767** with Otsu
segmentation. Validated against **three human annotators** who graded 150 leaves
(agreeing with each other at quadratic κ 0.63, the ceiling), the lesion-ratio grade
tracks their consensus at κ 0.30, rising to **0.47** after cross-validated
recalibration — a moderate proxy at ~74% of the human ceiling. Trained *jointly*
with the classifier, a severity-regression head becomes a genuine model output:
it reproduces the official mask almost perfectly (within-class ρ **0.957**) at no
classification cost, and beats both classical estimators on human agreement (ρ 0.47
vs Otsu 0.43). All lesion-area methods plateau near the human ceiling, because
severity is more than lesion area — reported honestly rather than inflated.

**Task 1 as the brief words it.** For "healthy vs diseased", the ensemble reaches
**99.96%** on the held-out test split — no error crosses the healthy/diseased
boundary. That figure falls to **80.5%** on field photographs, so even the coarse
decision the brief asks for is partly propped up by the benchmark. The honest
deployment path is domain adaptation: fine-tuning on field data recovers accuracy
from 16.1% to **55.9%**.

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
