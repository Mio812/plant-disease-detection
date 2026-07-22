# Working plan

Live state of the project. `docs/EXPERIMENTS.md` fixes *what* is being tested and
what would falsify it; this file tracks *where we are* and *what happens next*.
Update it as stages land, so nothing has to be reconstructed from memory later.

Last updated: 2026-07-20, during the `push.ps1` queue.

## Leaf-leakage rebuild (in progress)

`leaf-map.json` revealed that the random split leaked same-leaf duplicates: 74.7%
of the test set (among images with a known leaf) had a same-leaf twin in training,
so the 99.84% was partly reading back leaves already seen. `splits.py` now
partitions leaves within each class (0% leakage, verified). The whole matrix is
retraining under it via `rebuild.ps1`; leaky results are preserved in
`outputs_leaky/` for the inflation comparison.

Three defects found and fixed along the way:
- the split was leaking (74.7%) — the headline finding
- `.gitignore`'s unanchored `data/` had kept `src/data/` — including `splits.py`,
  the single source of truth — out of git since the project began
- `get_dataloaders` passed `len(base)`, so `ensemble.py` would have scored Task 1
  on the leaky split even after the fix; the int-fallback exists to make that kind
  of miss loud, and it was the seventh and last caller

Expectation, to be checked against results: the leak was constant across arms, so
every A-vs-B conclusion (H5, H8, McNemar) should survive; only the absolute
PlantVillage accuracies should fall. If a comparative conclusion flips, that is
itself a finding and gets recorded.

Leaky reference (to be replaced by honest numbers): ensemble 99.84 / binary 100.00;
resnet18 arms 99.52 / 99.26 / 91.70 (frozen) / 99.24 (seg) / 98.48 (gray).

## E26 landed and is validated (tonight)

Joint training makes severity a model output that reproduces the official mask
near-perfectly (within-class rho 0.957, reproduced exactly on a second run,
non-degenerate) at no classification cost (99.44%). The human cross-check is the
honest headline: on the 20 graded leaves in E26's test split, the head agrees with
human consensus at rho 0.47, beating Otsu 0.43 and the mask 0.41 — the best
estimator we have and a model output, but lesion-area severity plateaus near human
rho 0.47 because people grade by more than lesion area. n=20 is small, so the
ordering is indicative.

Report framing: E26 satisfies "expand the model" (severity is now a network output,
best-in-class, no mask at inference), reported honestly with the human ceiling.
Artifacts: outputs/joint_model.pth, joint_predictions.json, joint_severity.json.

## Tomorrow, first thing

**E26 — joint training (H17).** Script built and CPU-verified (`scripts/train_joint.py`;
unit checks + smoke test passed). Just launch on the free GPU:

    uv run python -m scripts.train_joint

Defaults: standard augmentation (strong ColorJitter would randomise the colour severity
needs), 20 epochs, 224px, severity_weight 10. ~50 min. Compare `within_class_rho_head`
to Otsu 0.78 and to E25's frozen probes (0.38 fine-tuned / 0.61 ImageNet) in
`outputs/joint_severity.json`. If it beats 0.78, H17 confirmed and severity becomes a
model output; if not, record it and Task 2 stands on the honest classical estimator.
`severity_weight`: at 10 the severity term is only ~3% of the loss (sev_mse is
inherently small on sqrt-ratio targets), so if within-class rho is weak, try
*higher* (30, 50, 100), not lower.
Also note `classification_accuracy` for any Task 1 trade-off.

Then: report (.docx) and PPT, with PlantDoc/073 as the discussion-level extension.

## Rule for this phase

**Finish the experiments before writing the report.** Every number that can still
move sits upstream of the prose, and rewriting the report once per landing stage
costs more than waiting. The report, the slides and the notebook execution all
happen after the matrix is closed.

The core argument is already complete and will not change: E3, E5, E7, E10, E15,
the error decomposition and the E11/E18 curve. What remains either refines a
number or satisfies a brief requirement literally — none of it can overturn a
conclusion.

## Running now

| Stage | Experiment | Decides |
|---|---|---|
| `train_frozen_bg_random` | E15 | completes the 2x2 (frozen x background randomisation) |
| `finetune_baseline` | E11 | the 128px adaptation comparison point |
| `eval_arm_bg_control` / `_bg_random` | E8, E16 | **whether E8's +0.42 survives on 2,525 images** |
| `eval_arm_frozen_ctrl` / `_frozen_bg` | E15, E16 | the frozen arms' decomposition |
| `eval_arm_hier` | E17, E16 | **H9 — crop accuracy against `eval_arm_bg_random`, same basis** |

Nothing else may touch `src/` or `scripts/` until these finish.

## What the project is

**090 is the whole project.** Deliver both its tasks excellently on PlantVillage,
then add PlantDoc as a bonus. PlantDoc is kept but subordinate — a Discussion-level
generalization check, never the headline, done only after 090 is complete. Do not
frame this as two projects. (User directive, 2026-07-20.)

Its two tasks:

1. a CNN that categorises leaves into **healthy and diseased** classes (and the
   specific disease)
2. the model **expanded** to estimate **disease severity from image features**

### Definition of done — "perfect 090"

**Task 1 — classification**
- [ ] three CNN baselines (custom_cnn, resnet18, mobilenet_v2) + soft-voting
      ensemble, on the honest leaf-grouped split *(rebuilding now)*
- [ ] healthy/diseased binary **and** 38-way specific disease, reported together
- [ ] all three variants exercised: color (primary), grayscale, segmented
- [ ] per-class precision/recall shown, so the 36x imbalance is demonstrated not to
      sink rare classes (rather than asserting it)
- [ ] honest numbers stated beside the leaky ones, leakage quantified

**Task 2 — severity from image features**
- [ ] a severity **head on the CNN** (E25/E26) — this is what literally satisfies
      "expand the model", even where the classical Otsu ratio grades finer
- [ ] the estimator reported honestly: presence AUC, within-class rho vs Otsu
- [x] **E14** — 150 human grades, two annotators. Done. Inter-annotator κ 0.72;
      lesion-ratio vs consensus ρ 0.47 / κ 0.29 (weak proxy, honest RQ3 answer).
      Raw grades in outputs/annotation/grade{1-4}.csv (A) and grades{1-4}.csv (B) —
      irreplaceable, back these up (outputs/ is gitignored)

**Core 090 analysis (exploratory + discussion, not bonus)**
- [ ] leaf-leakage / test integrity — the split fix and its inflation number
- [ ] residual-leakage check (post-rebuild): embed all images with a frozen
      ImageNet backbone, calibrate a "same-leaf" cosine threshold on the pairs the
      leaf-map already groups, then count test images with an out-of-group near
      duplicate in train. Quantifies what the map missed among the 24% singletons.
      Built and logic-verified in scratchpad (`near_dup_check.py`); GPU pass, run
      once the rebuild frees the machine
- [ ] E3 border-pixel probe, E5 Grad-CAM — what the classifier attends to
- [ ] E10 grayscale colour ablation
- [ ] class-distribution / imbalance characterization

**Only once the above is solid — the PlantDoc bonus**
- [ ] E6 zero-shot field, E11/E18 adaptation, E16 decomposition — as a Discussion
      generalization check ("does the lab number hold in the field? no, and here is
      why"), explicitly a bonus

Weighting: Task 2 is half of 090 and still carries the least evidence (E14 at 0/150),
so it is not allowed to trail classification the way it currently does.

## Order of execution

**Close both core tasks on PlantVillage before touching PlantDoc again.** 090 is
the brief; its dataset is the three raw variants. Field work is an extension and
waits. Run strictly in sequence; each phase gates the next.

### Task 1 — closed

| variant | 38-way | binary |
|---|---|---|
| color, ensemble | **99.84** | **100.00** |
| segmented, trained on it | 99.24 | — |
| grayscale, trained on it | 98.48 | — |

All three raw parts used. 13 errors remain out of 8,145 (0.160%): eleven are
within-crop disease confusions that are genuinely ambiguous — Corn Cercospora
against Northern Leaf Blight, Tomato early against late blight — and two are
healthy-to-healthy across crops. **No healthy image is ever called diseased or the
reverse**, which is why the binary number is exactly 100.00. There is no headroom
here worth spending time on; the residue is dataset noise.

### Task 2 — the open half

**Phase 0 — while the queue is live.** No edits to `src/` or `scripts/`.
- [x] pre-register H15/E24 (c4371ce) and H16/E25
- [x] fix the order in this file
- [ ] draft the lesion-ratio caching pass in the scratchpad, ready to install

**Phase 1 — done.** The nine-stage queue finished in 162 min.
- [x] five `eval_arm_*` artefacts read on the 2,525 basis
- [x] **H8 corrected to confirmed** (046cd36, b5af615). The n = 236 "tie" was
      underpowered; freezing wins, crop p = 2.1e-08
- [x] **H5 corrected to confirmed.** Recorded as null on +0.42 at n = 236; it is
      p = 0.0027 on 38-way and p = 5.2e-05 on crop once measured on the
      pre-specified basis with the pre-specified test
- [x] **H9 magnitude rejected** — crop +1.74, p = 0.014 raw, fails Bonferroni
- [x] deferred patches applied (39da42a); float32 measured at 4.50 → 2.25 MiB,
      and the generator switch buys nothing over the stream-preserving cast
- [x] `--save-predictions`, `mcnemar()`, and the `paired` probe (b5af615)

**Phase 2 — finish task 2 on PlantVillage. Everything else waits.**
- [x] cache lesion ratios for all 54,305 images from the official masks (54,304 cached)
- [x] verify the target is severity and not an artefact: healthy median 0.0216 vs
      diseased 0.1006, corpus AUC 0.8324, lowest-median classes all healthy
- [ ] **E25** severity head on the trained backbone, scored against its floor
      (Otsu) and ceiling (official mask) on the same test split
- [ ] if a linear head on frozen features cannot recover severity, try an MLP head
      and joint fine-tuning **before** concluding the features lack it
- [ ] **E23** better unsupervised mask — raises the deployable floor, PlantVillage-only
- [ ] **E14** the moment gradings return: `annotate --action merge`, then
      `audit --probe severity-validate`. This is the only evidence severity is
      *trustworthy* rather than merely correlated

**Phase 3 — the 073 extension, once both core tasks are closed.**
- [ ] **E24** frozen-backbone ladder (drafted, CPU-verified, waiting)
- [ ] **E22** direct binary head — no headroom on PlantVillage, where binary is
      already 100.00; it is a field-accuracy arm and belongs here
- [ ] E19 scale probe, E21 capture-condition strata

**Phase 5 — deliverables, once the matrix is closed.**
- [ ] 073 stage labels in `configs/experiments.yaml`
- [ ] notebook extended to E15–E18 and E24/E25, cell ids added
- [ ] report `.docx`, then the UNSW PPT template filled as-is
- [ ] `nbconvert --execute --inplace` last, so outputs are visible

## Blocked on people

**E14, 0 of 150 graded.** Packets are built and validated at
`outputs/annotation/annotator_{1..4}` (38/38/38/36). Gates both the report and the
notebook, and no amount of compute substitutes for it. On return:
`annotate --action merge` then `audit --probe severity-validate`.

## Decisions already taken — do not re-litigate

- **Baseline for the accuracy-improvement claim.** Report both: +38.56 pp against
  the baseline ensemble (17.37, the thing literally called a baseline) and
  +31.78 pp against the adapted arm's own zero-shot (24.15, which isolates
  adaptation). Never quote only the tightest.
- **Do not tune to cross a threshold.** The project spends E3/E5/E15 proving that
  benchmark-chasing produces hollow numbers; doing it ourselves would break the
  argument. TTA and adapted ensembles are separate reported lines. Improving
  `ft_full` is engineering and stays outside the matrix.
- **Severity bands realigned before grading** (9b0017b), so a rubric-following
  annotator scores as agreeing.
- **McNemar declared before computing** (a023841). The design is paired; comparing
  independent Wilson intervals was the wrong instrument from the start.
- **Zero-shot and adapted never share a table row.** 236 and 2,525 never mix.
- **PlantDoc is the instrument RQ2 requires**, never a second project.
- **Binary is the brief's Task 1 metric**, reported next to the 38-way headline,
  never instead of it.

## Do not forget

- **Never edit `src/` or `scripts/` while a run is live.** Windows workers
  re-import at each epoch boundary; this has already destroyed two multi-hour runs
- `evaluate.py` needs `--image-size` for every 224 arm, or it silently reports ~13%
- At most two training jobs at once
- Close `plant_disease_detection.ipynb` in VSCode *before* regenerating it —
  autosave clobbers external edits
- **E18's 100-shot point uses 92.7% of PlantDoc train and exhausts 20 of 28
  classes.** It is not independent evidence of saturation; quote the curve against
  images actually used (137 / 542 / 1,342 / 2,124 / 2,291)
- `outputs/ft_baseline.pth` is an orphan from a run that died at 18:33;
  `finetune_baseline` overwrites it and never reads it
- `logs/push_run.log` is untracked and probably wants a `.gitignore` entry
- nbformat warns cells lack `id` fields, and says it becomes a hard error later
