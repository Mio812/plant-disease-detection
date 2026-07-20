# COMP9444 25T1 — Project 090, Automatic Plant Disease Detection

## What this project argues

Not "we got 99.8% on PlantVillage". The argument is that **99.8% on PlantVillage is
mostly capture bias**, and the project's job is to prove that with controls rather
than assert it. Every headline number is paired with a field number; a lab number
on its own is not a result.

Read `docs/EXPERIMENTS.md` before changing anything — it states the hypotheses
(H1–H9), the experiment matrix (E1–E18), and the conditions under which each
conclusion would be **falsified**. Those falsification conditions were written
before the results came in and must not be quietly edited afterwards. H8 already
failed its prediction and is recorded as "partly rejected"; keep that honesty.

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
