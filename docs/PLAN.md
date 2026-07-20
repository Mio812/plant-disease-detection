# Working plan

Live state of the project. `docs/EXPERIMENTS.md` fixes *what* is being tested and
what would falsify it; this file tracks *where we are* and *what happens next*.
Update it as stages land, so nothing has to be reconstructed from memory later.

Last updated: 2026-07-20, during the `push.ps1` queue.

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

## What the project is graded against

090 is the core and everything is ordered by it. Its two tasks:

1. a CNN that categorises leaves into **healthy and diseased** classes
2. the model **expanded** to estimate **disease severity from image features**

073 is an extension: it tests whether task 1 survives outside the lab. Useful,
never co-equal.

Weighting matters here. Task 2 is half the brief and currently carries three
experiments (E12, E13, E14) against roughly twenty on classification, and E14 has
no progress at all. Work is ordered to correct that, not to keep deepening the
half that is already strong.

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
