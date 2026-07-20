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

## Queue, in order

1. **073 labels** in `configs/experiments.yaml` so the numbering is continuous
   (docs side already done: `EXPERIMENTS.md` §5, `CLAUDE.md`)
2. **Deferred patches** — float32 in `random_background`, `--num-workers` for
   `scripts/finetune.py`. After *all* training, not merely an idle GPU
3. **Per-image predictions** in `evaluate.py`, then McNemar on the paired arms
4. **E22** — direct binary head. Literal compliance with brief Task 1
5. **Notebook** — extend `tools/build_notebook.py` to E15–E18, add cell ids
6. **E23** — better leaf mask, closes the 0.767 → 0.868 severity gap
7. **E19 / E21** — scale probe, robustness strata. If time permits
8. **Report, slides, notebook execution** — once the above is closed

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
