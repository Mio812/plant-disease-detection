# Working plan

Live state of the project. `docs/EXPERIMENTS.md` fixes *what* is being tested and
what would falsify it; this file tracks *where we are* and *what happens next*.

Last updated: after the leaf-grouped rebuild, E26, E21 and the figure pass.

## What the project is

**090 is the whole project.** Both of its tasks are delivered on PlantVillage;
PlantDoc enters afterwards as a subordinate generalisation check, never the
headline. Do not frame this as two projects.

1. a CNN that categorises leaves into **healthy and diseased** classes (and the
   specific disease)
2. the model **expanded** to estimate **disease severity from image features**

## Status — both tasks complete, on honest numbers

**Task 1 — classification.** Three CNN baselines (98.69 / 99.33 / 99.46) plus a
soft-voting ensemble at **99.53%** over 38 specific diseases and **99.96%** on
healthy-vs-diseased, over 8,215 held-out images. All three PlantVillage variants
exercised (colour 99.53, segmented 98.66, grayscale 97.76). Per-class recall is
reported, so the 36:1 imbalance is shown not to sink the rare classes.

39 errors remain. Most are within-crop look-alikes; **three cross the
healthy/diseased boundary, two of them diseased leaves called healthy** — which is
why binary is 99.96 and not a perfect 100.

**Task 2 — severity.** Lesion-area ratio separates healthy from diseased at
AUC 0.868 (official masks) / 0.767 (Otsu). Validated against **three annotators**
who graded 150 leaves and agree with each other at quadratic κ **0.634** (the
ceiling); the grade tracks their consensus at κ 0.30, rising to **0.472** after
cross-validated recalibration. Trained *jointly* with the classifier (E26), a
severity head reproduces the official mask at within-class ρ **0.957** with no
classification cost (99.44%), and beats both classical estimators on human
agreement (ρ 0.47 vs Otsu 0.43). All lesion-area methods plateau below the human
ceiling — reported as the honest limit rather than inflated.

**Test-set integrity.** `leaf-map.json` showed a random split leaks same-leaf
duplicates: 74.7% of the test set had a twin in training. The split is now by leaf
(0% leakage, verified); adopting it moved the ensemble 99.84 → 99.53. A perceptual
re-check bounds the residual the metadata missed at ≤10.7%, an over-estimate.

**PlantDoc extension.** Zero-shot 16.10% (236) / 13.94% (2,525); the bottleneck is
crop identification (41.39% for the best arm), not diagnosis. Adaptation recovers
16.10 → **55.93%**. E21 shows field accuracy is flat across capture-quality bins,
so the failure is the domain gap, not photograph quality.

## Deliverables

- [x] codebase, config-driven and reproducible from one seed
- [x] notebook, regenerated and executed with visible outputs
- [x] report `.docx` — six figures embedded with captions
- [x] slides — honest numbers and the thesis, on the UNSW template, with the
      confusion matrix, the lab-to-field collapse and the adaptation curve
- [x] result artefacts and the 150 human grades committed as evidence
- [ ] team/presenter placeholders in the deck — for the team to fill
- [ ] a human read-through of the report and deck in Word/PowerPoint

## Open, optional

None of these changes a conclusion; they are refinements only.

- **E24** frozen-backbone ladder (`scripts/probe_backbones.py`, CPU-verified, never
  run) — tests whether field accuracy tracks backbone quality (H15)
- **E19/E20** leaf-scale mismatch (H10) — the one untested input-side lever
- **E22** direct binary head (H12), **E23** better unsupervised mask (H13)
- `experiments.yaml` evaluates with fixed weights `[0.2, 0.5, 0.3]` while
  `ensemble.py` tunes on validation and gets `[0.5, 0.15, 0.35]` (99.54 vs 99.53).
  Everything reports the former consistently; align them if the "validation-tuned"
  wording needs to be literal.

## Rules that still apply

- **Never edit `src/` or `scripts/` while a run is live** — Windows workers
  re-import at epoch boundaries; this has destroyed multi-hour runs before.
- `evaluate.py` needs `--image-size` for every 224 arm, or it silently reports ~13%.
- At most two training jobs at once.
- Close `plant_disease_detection.ipynb` in VSCode *before* regenerating it —
  autosave clobbers external edits.
- Lab and field numbers are always reported together; the 236- and 2,525-image
  bases are never mixed in one comparison.
- Quote the `evaluate.py` ensemble (99.53) everywhere, not `ensemble.py`'s 99.54,
  so per-disease figures and headline agree.

## Decisions taken — do not re-litigate

- **Dual baseline for the improvement claim.** Report both +39.83 pp over the
  baseline ensemble (16.10) and the adaptation delta; never quote only the flattering one.
- **Do not tune to cross a threshold.** The project spends E3/E5/E15 proving that
  benchmark-chasing produces hollow numbers; doing it ourselves would break the argument.
- **Severity bands were realigned to the rubric before grading**, so a
  rubric-following annotator scores as agreeing.
- **McNemar was declared before any paired statistic was computed** — the design is
  paired, and independent Wilson intervals were the wrong instrument.
- **PlantDoc is the instrument RQ2 requires**, never a second project.
