# COMP9444 25T1 — Project 090, Automatic Plant Disease Detection

## The project in one page

**The brief.** Project 090 asks for a CNN that (i) classifies leaf images as healthy
or diseased and (ii) estimates disease severity from image features, so as to *"aid
farmers and agricultural experts in timely intervention"*. Deliverables are a
codebase, a Jupyter notebook with visible outputs, a summary report (.docx) and a
presentation that must fill the **UNSW-provided template** as-is. Team of four.

**The twist that makes it a real project.** The stated purpose is a *deployment*
claim. PlantVillage — the dataset the brief points at — is 54,305 photographs of
single detached leaves, laid flat on uniform backgrounds under studio lighting.
Any modern CNN scores above 99% on it. That number does not answer the brief's own
question, so the work splits into three:

- **RQ1** — Can a CNN classify PlantVillage leaves accurately? *(brief, Task 1)*
- **RQ2** — Does that accuracy mean it would work for the farmer the brief
  describes, i.e. on real field photographs?
- **RQ3** — Can severity be estimated from image features, and is it trustworthy?
  *(brief, Task 2)*

RQ2 is not scope creep; it tests whether the brief's stated aim is met.

**Data.**

| | Images | Classes | Role |
|---|---|---|---|
| PlantVillage `color` | 54,305 | 38 (14 crops) | train / val / test, seed 42, 70-15-15 |
| PlantVillage `grayscale` | 54,305 | 38 | colour-cue ablation (E10) |
| PlantVillage `segmented` | 54,305 | 38 | leaf masks for E4/E9, background randomisation, severity, Grad-CAM |
| PlantDoc | 2,525 usable | 27 mapped | **never trained on** — external field validation |

The three PlantVillage variants are the same leaves, so the split is index-identical
across them. PlantDoc is web-scraped in-the-wild photography: whole plants, varying
scale, overlapping foliage, real backgrounds.

**Models.** Three baselines — a from-scratch CustomCNN, ResNet-18 and MobileNet-V2 —
plus a soft-voting **ensemble** whose weights are grid-searched on validation and
never see test. Later arms add background randomisation, a frozen-backbone probe,
and a factorised crop-then-disease head.

**What the project actually argues.** Not "we got 99.8%". The argument is that
**99.8% on PlantVillage is mostly capture bias**, demonstrated with controls rather
than asserted, and that the honest deployment story is a domain-adaptation one.
Four independent probes agree, and the strongest is a direct control: training only
19,494 parameters (0.17% of the network) reaches 91.70% in the lab and the *same*
24.15% in the field as training all 11.2M. The 7.8 lab points that full fine-tuning
buys are worth nothing outside the benchmark.

**Why PlantDoc is in a PlantVillage project.** Project 090 names PlantVillage and
asks for classification plus severity. Its stated purpose — timely intervention by
farmers — is a deployment claim, and PlantVillage cannot test one: it is studio
photography of detached leaves. PlantDoc is field photography of whole plants, and
it is the only ready-made instrument for that test. It therefore enters as **the
instrument RQ2 requires**, not as a second project.

PlantDoc happens to be the dataset of a separate brief (073), so the field work is
held to that brief's requirements as well, and `docs/EXPERIMENTS.md` §5 records the
mapping so coverage is traceable. Two rules follow. **Never describe this as having
completed two projects** — it is one project that took its own brief's stated aim
seriously. And both bullet points of 090's Task list stay first-class: healthy vs
diseased, the specific disease, and severity.

Read `docs/EXPERIMENTS.md` before changing anything — it fixes the hypotheses
(H1–H9), the experiment matrix (E1–E18) and the conditions under which each
conclusion would be **falsified**, all written before the results arrived. Do not
quietly edit a falsification condition afterwards. H8 already failed its prediction
and is recorded as "partly rejected"; keep that honesty. Every headline number is
paired with a field number; a lab number alone is not a result.

## Layout

```
configs/experiments.yaml   declarative stage matrix; `produces` makes it resumable
docs/EXPERIMENTS.md        hypotheses, controls, falsification conditions
src/data/                  splits (single source of truth), variants, PlantDoc mapping
src/models/                backbones, ensemble, hierarchical crop-then-disease head
src/training/              fit loop, strong augmentation, background randomisation
src/evaluation/            metrics, ONE inference path, plots
src/audit/                 background probe, Grad-CAM, severity
scripts/                   9 entry points, all `python -m scripts.<name>`
push.ps1 / run.ps1         sequential drivers
```

Two invariants that were expensive to establish — do not undo them:

- `src/data/splits.py` is the **only** place the train/val/test split is computed.
  It was duplicated in four files and they drifted.
- `src/evaluation/inference.py` is the **only** inference path. There were six
  copies of "load a checkpoint and softmax it" and they disagreed.

## Traps that have already cost real time

**Never edit files under `src/` or `scripts/` while a training job is running.**
Windows DataLoader workers use spawn and re-import the module at every epoch
boundary. A half-written file is a `SyntaxError` inside a worker and kills the run.
This destroyed two multi-hour jobs. Wait for an idle window.

**`evaluate.py` needs `--image-size`.** The config default is 128 but every 224 arm
was trained at 224. Evaluating a 224 checkpoint at 128 silently reports ~13% instead
of ~24%. The stages pass it explicitly; keep it that way.

**Do not run more than two training jobs concurrently.** `num_workers: 10` times
three windows is 30 spawned Windows processes and it OOMs. Use `--num-workers` to
bound it, or run sequentially via `push.ps1`.

**`data_distribution_for_SVM/` is not the split.** It has more test images than
train and exists for the SVM baseline. `docs/EXPERIMENTS.md` explains this.

**PlantDoc filenames:** ~96 files contain `?` from URL query strings and are invalid
on NTFS; they are excluded by sparse-checkout. Some JPEGs are truncated, hence
`ImageFile.LOAD_TRUNCATED_IMAGES = True` in `src/data/plantvillage.py`.

**PowerShell `Tee-Object` writes UTF-16**, so `grep` on logs fails. `scripts/status.py`
decodes it.

## Conventions

- Comments explain *why*, never *what*. The code says what it does. Keep them sparse.
- Every claim in the report traces to a JSON file in `outputs/`; the notebook reads
  those files rather than recomputing, so it stays runnable.
- New experiments go in `configs/experiments.yaml` with an `experiment:` field
  pointing at a hypothesis in `docs/EXPERIMENTS.md`. An arm with no hypothesis and
  no control does not belong in the matrix.
- Zero-shot and adapted (E11/E18) numbers **never share a table row**. Adaptation
  touches target labels; conflating them would overstate the result.
- Report field accuracy over all 38 classes as the headline, with the 27-class
  restricted figure alongside — never instead.

## Verifying before spending GPU time

The user has repeatedly, and correctly, asked for verification before committing
hours of compute. The pattern that works:

1. `python -m scripts.run_all --dry-run` — check the commands expand correctly.
2. Unit-check the new mechanism directly (parameter counts, BatchNorm modes,
   whether a distribution sums to 1, gradient finiteness).
3. Smoke-test on a tiny synthetic ImageFolder before the real dataset.
4. Confirm existing tags are unchanged so cached checkpoints stay valid.
