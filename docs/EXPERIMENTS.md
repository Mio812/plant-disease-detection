# Experimental Plan

COMP9444 25T1 — Project 090, *Automatic Plant Disease Detection Using Computer Vision*.

This document fixes the research question, the hypotheses, and the experiment
matrix **before** any result is quoted, so that every run in `experiments.yaml`
exists to test a stated hypothesis rather than to fill a table.

## 1. Research question

The brief asks for a CNN that (i) classifies leaves as healthy or diseased and
(ii) estimates disease severity from image features, in order to *"aid farmers
and agricultural experts in timely intervention"*. That stated purpose is a
deployment claim, and it splits the work into three questions:

- **RQ1** — Can a CNN classify PlantVillage leaves accurately? *(brief, Task 1)*
- **RQ2** — Does that accuracy mean the system would work for the farmer the
  brief describes, i.e. on real field photographs?
- **RQ3** — Can severity be estimated from image features, and is the estimate
  trustworthy? *(brief, Task 2)*

RQ2 is not an extension of the brief; it tests whether the brief's own stated
aim is met. It is also where the marks for *Discussion*, *Results* and
*comparison with prior work* are earned.

## 2. Hypotheses

| ID | Hypothesis | Status |
|----|------------|--------|
| H1 | Standard CNNs exceed 99% on the PlantVillage test split. | confirmed |
| H2 | Much of that accuracy comes from capture bias in the background, not from leaf pathology. | confirmed |
| H3 | Accuracy therefore collapses on real field photographs. | confirmed |
| H4 | The collapse is a *learned shortcut*, not a test-time statistics shift, so test-time fixes will not repair it. | confirmed |
| H5 | Removing the shortcut during training improves field transfer, at a small cost in lab accuracy. | **open — needs training** |
| H6 | Lesion-area ratio is a valid severity signal, and official leaf masks beat Otsu segmentation. | confirmed |
| H7 | Field accuracy is limited by crop identification, not by disease diagnosis. | confirmed |
| H8 | Full fine-tuning on PlantVillage degrades the pretrained features that transfer to field images, so a frozen backbone transfers better. | **partly rejected** — it ties, it does not win |
| H9 | Making crop an explicit subproblem raises field accuracy, because the crop term is the binding constraint. | open |
| H10 | The residual field gap is a *scale* mismatch. PlantVillage leaves already fill 47.5% of the frame and both augmentation recipes only ever enlarge them, so a leaf at field apparent size falls outside the training support entirely. | open |
| H11 | Strong photometric augmentation buys measurable robustness to field capture variation, so accuracy degrades less across the lighting and sharpness tails than it does for standard augmentation. | open |
| H12 | Task 1 is healthy-vs-diseased, which does not require species identification — the very thing H7 shows is the field bottleneck. Training the binary objective directly should therefore transfer better than collapsing a 38-way model's predictions. | open |
| H13 | The Otsu-versus-official-mask AUC gap is segmentation error, not a ceiling on the lesion-ratio feature, so a better unsupervised mask recovers part of it. | open |
| H14 | Background randomisation buys little zero-shot, but leaves features that *adapt* better, so its advantage grows under supervised adaptation rather than disappearing. | open |

### The clearest single result: 574x the parameters, no field gain

E15 trains only the 19,494-parameter head and leaves all 11.2M backbone weights
at their ImageNet values. The matched full fine-tune updates everything.

| Arm | Trainable | PlantVillage | PlantDoc |
|---|---|---|---|
| Full fine-tune, strong aug (E8 control) | 11,196,006 | 99.52% | 23.73% |
| Frozen backbone, strong aug (E15 control) | 19,494 (0.17%) | 91.70% | 24.15% |

Fine-tuning 574x more parameters buys 7.8 points on PlantVillage and nothing at
all in the field. H8 predicted the frozen arm would *win*; it ties, so H8 is
recorded as partly rejected. The tie is the stronger statement: it isolates those
7.8 points as entirely benchmark-specific, using a direct control rather than the
indirect evidence of E3 and E5. The two accuracies are decoupled.

### What the field accuracy is actually made of

Reporting a single 24% hides the failure mode. Decomposing the best zero-shot arm
(ResNet-18, strong augmentation, `p = 0.7`, 224px) on the 236-image PlantDoc test
split:

| Level | Accuracy | Chance |
|---|---|---|
| All 38 classes | 24.15% | 2.6% |
| Restricted to the 27 reachable classes | 27.54% | 3.7% |
| Crop species only | 46.19% | 7.1% |
| Disease, given the crop was right | 52.29% | ~33% |
| Healthy vs diseased | 74.58% | 50% |

`0.4619 x 0.5229 = 0.2415`, so the two factors account for the whole number. Once
the species is right the diagnosis is 1.6x chance, which is weak but real; the
species itself is right less than half the time. The bottleneck is therefore
recognising the plant, not recognising the disease.

Every figure in that table is on n = 236. `scripts/train.py` prints only the total
for that split and never the decomposition, so no arm's crop accuracy can be read
off a training log. E16 re-scores each 224 arm on all 2,525 field images, and
cross-arm comparisons — E17 against E8 above all — are made on that basis alone.
The two are never mixed.

This is consistent with H2 rather than a separate finding. PlantVillage shows one
detached leaf, centred, flat, filling the frame; PlantDoc shows whole plants at
varying scale with overlapping foliage. Leaf outline — the main species cue —
survives none of that. It also explains why background randomisation buys so
little (E8): it replaces the background but leaves the single-leaf, centred,
frontal composition intact, and that composition is the deeper bias.

H8 follows directly. If PlantVillage rewards shortcut features, then updating all
11.2M weights on it should actively damage the ImageNet features that would have
transferred, and training only the 19K-parameter head should transfer better.

### Why we build our own split

The PlantVillage repository ships `data_distribution_for_SVM/`, a train/test
directory pair that is easy to mistake for a canonical split. It is not: it holds
19,300 images (35% of the corpus) in 38 numerically-named classes, with **more
test images than training images** (10,547 vs 8,751). That layout suits the SVM
feature-extraction baseline it was built for, not CNN training.

We therefore use all 54,305 colour images under one seeded 70/15/15 split
(`src/data/splits.py`), shared by every model, every dataset variant and every
audit, and never re-derived anywhere else in the codebase.

### Which dataset variants are used

| Variant | Used by |
|---------|---------|
| `color` | all training and evaluation unless stated otherwise |
| `grayscale` | E10, the colour-cue ablation |
| `segmented` | E4 and E9 directly; also supplies the leaf masks for background randomisation (E8), the severity estimator (E12/E13) and the Grad-CAM leaf-attention metric (E5) |

## 3. Experiment matrix

Every PlantVillage number uses one fixed split (seed 42, 70/15/15, identical
images across all dataset variants). Field numbers carry Wilson 95% intervals.

| ID | Experiment | Tests | Control / confound | Status |
|----|------------|-------|--------------------|--------|
| E1 | Three baselines: custom CNN, ResNet-18, MobileNet-V2 | H1 | shared pipeline, identical split | done |
| E2 | Soft-voting ensemble, weights tuned on validation | H1 | weights never see the test split | done |
| E3 | Classify from 8 background border pixels only | H2 | chance = 1/38 = 2.6% | done |
| E4 | Re-score models with the background removed (`segmented`) | H2 | **confounded** by an unseen black-background domain → controlled by E9 | done |
| E5 | Grad-CAM mass inside the leaf mask | H2 | leaf area fraction is the baseline | done |
| E6 | Zero-shot on PlantDoc field images | H3 | model never saw any PlantDoc image | done |
| E7 | Test-time fixes: hflip TTA, AdaBN | H4 | negative result is informative | done |
| E8 | Background randomisation, p ∈ {0.0, 0.7, 1.0} | H5 | **p = 0.0 is the control**: isolates de-shortcutting from stronger augmentation | open |
| E9 | Train on `segmented` | H5, control for E4 | removes E4's domain-shift confound | open |
| E10 | Train on `grayscale` | colour-cue ablation | third variant named in the brief | open |
| E11 | Few-shot fine-tune on PlantDoc train | supervised ceiling | **reported separately** — touches target labels | partial |
| E12 | Leaf segmentation vs official masks (Dice) | H6 | official mask is ground truth | done |
| E13 | Lesion ratio separates healthy vs diseased (ROC-AUC) | H6 | needs no manual labels | done |
| E14 | Ordinal grade vs manual annotation (ρ, MAE, κ) | H6 | 150 leaves, graded by the team; bands fixed to the rubric before grading | open |
| E15 | Frozen backbone vs full fine-tune, crossed with `p ∈ {0.0, 0.7}` | H8 | 2x2 factorial: separates both main effects and their interaction | open |
| E16 | Crop / disease / restricted decomposition of every 224 arm | H7 | scored on all 2,525 PlantDoc images, not the 236-image split | open |
| E17 | Factorised crop-then-disease head | H9 | matched to E8 in every respect but the head | open |
| E18 | Adaptation curve at 5 / 20 / 50 / 100 / all shots | supervised ceiling | one point is not a curve; shows how much field data is actually needed | open |
| E19 | Score PlantDoc at several test-time zoom factors | H10 | the current single 1.0x resize is the control; E7 varied flips and BatchNorm but never scale | open |
| E20 | Train with the leaf composited at 20-60% of the frame | H10 | matched to E8 `p = 0.7` in every respect but leaf scale | open |
| E21 | Field accuracy stratified by luminance, contrast and sharpness | H11 | standard-augmentation baseline is the control; Wilson interval per bin | open |
| E22 | Two-way healthy/diseased head, trained directly | H12 | matched to E8 `p = 0.7` in every respect but the output space; control is binary collapsed from the same arm's 38-way predictions | open |
| E23 | Alternative unsupervised leaf mask | H13 | current `_leaf_mask` heuristic is the control, scored by Dice *and* by the severity AUC it produces | open |

## 4. What would falsify the conclusions

- If E3 scored near 2.6%, and E5 showed attention concentrated on the leaf,
  H2 would be rejected and the 99.8% would be taken at face value.
- If E8 with `p = 0.0` matched `p = 0.7` on PlantDoc, the gain would be
  attributable to augmentation alone and H5 would be rejected. **This is what we
  observe**: 23.73% vs 24.15%, a gap far inside the ±5.5pp interval at n = 236.
  E15 and E16 re-score both arms on all 2,525 images to decide it properly.
- If the frozen backbone transferred *worse* than the full fine-tune, H8 would be
  rejected and the low field accuracy would have to be attributed to the domain
  gap alone rather than to fine-tuning damaging transferable features.
- If E9 (trained on `segmented`) scored poorly *in-domain*, the E4 collapse
  would be explained by loss of information rather than loss of a shortcut.
- If E13's AUC were near 0.5, the severity signal would be meaningless.
- If accuracy is flat across test-time zoom factors (E19) *and* E20's crop accuracy
  fails to beat E8 by more than the Wilson interval on the 2,525 basis, H10 is
  rejected: the compositional gap cannot be synthesised from detached leaves, and
  only real field data closes it.
- If every arm degrades at the same rate across the capture-quality bins (E21),
  H11 is rejected. 073-2 is then reported as *evaluated under* natural variation
  rather than *robust to* it, in the same way E7 and E8 are reported.

## 5. Mapping to the brief and the rubric

| Deliverable | Covered by |
|-------------|-----------|
| Brief Task 1 — healthy/diseased classification | E1, E2 (binary accuracy 100.00%) |
| Brief Task 2 — severity from image features | E12, E13, E14 |
| Brief — colour, grayscale and segmented variants | E10, E9, E4 |
| Rubric — Exploratory analysis (3) | E3, class distribution, severity distributions |
| Rubric — Models and methods (3) | E1, E2, E8, E9, E10 |
| Rubric — Results (3) | E1–E11, comparison with Mohanty/Ferentinos/Singh |
| Rubric — Discussion (2) | E4–E7 limitations, E14 calibration |

### Coverage of the added field-data requirements

PlantDoc is not a second project. It enters because RQ2 needs an instrument, and
it is the only ready-made one. It does, however, originate in its own brief, so
the field work is held to that brief's requirements as well. The mapping is
recorded here so coverage is traceable rather than incidental.

| Requirement | Met by | Evidence |
|---|---|---|
| Accuracy improvement over baseline methods (>31%) | E11 / E18, full-data adaptation | 17.37 → 55.93 = **+38.56 pp** over the baseline ensemble; **+31.78 pp** against the adapted arm's own zero-shot. Both n = 236 |
| Robustness to lighting, growth stage and symptom variation | E21, with E8 supplying the mechanism and E10 bounding the colour dependence | open — E21 |
| Lightweight enough for mobile or edge deployment | E15 with the efficiency probe | a frozen backbone plus a 19,494-parameter head is **78 KB per crop** against a 44.9 MB model, at no measured field cost (24.15% either way) |

Two qualifications. The source paper's own wording is an *increase in classification
accuracy*, and its baseline is a model trained on PlantVillage and tested on
PlantDoc, so both readings above are reported rather than whichever is larger.
And PlantDoc carries no growth-stage labels: E21 stratifies capture conditions
only, and growth stage is recorded as a limitation instead of being proxied by
something invented.

## 6. Reporting protocol

1. Lab and field accuracy are always reported **together**; a lab number alone
   is not a result.
2. A drop in PlantVillage accuracy when the shortcut is removed is expected and
   is *not* a regression — it is the price of generalisation.
3. E11 never shares a row with zero-shot numbers.
4. `ft_robust` (48.73%) and `ft_baseline` start from checkpoints differing in
   resolution, augmentation *and* background randomisation, so their gap is not
   evidence for any one of them. `ft_aug_only` differs from `ft_robust` in
   `p_random` alone and is the arm the E11 claim rests on.
5. Field accuracy is reported over the full 38-class output space as the headline,
   with the 27-class restricted figure alongside it. Restriction assumes the
   deployment knows which crops are planted, which is realistic but is an
   assumption, so it never replaces the unrestricted number.
6. PlantDoc is web-scraped: a small number of its JPEGs are truncated, and PIL is
   configured to load them rather than abort. Roughly 4% of the repository's file
   names are also invalid on NTFS and are excluded on Windows.
7. Known threats to validity: PlantDoc carries label noise;
   its train/test splits are known to differ in content; our field evaluation is
   236 images (±5 points), so the full-dataset variant is also reported.
8. Two arms scored on the *same* images are compared with **McNemar's test on the
   discordant pairs**, not by asking whether two independent Wilson intervals
   overlap. The design is paired and always was, so independent intervals were
   the wrong instrument from the start and threw away most of the power. This is
   declared here *before* any paired statistic is computed, and the outcome is
   reported whichever way it falls — including if it leaves E8 exactly as null as
   the intervals did. Wilson intervals stay for single-arm accuracies, where they
   are the right tool. Requires per-image predictions, which `evaluate.py` does
   not yet persist.
9. The adapted arms consumed PlantDoc train, so **236 images is the only legal
   test set they have**. Their comparisons can never be tightened the way
   `eval_arm_*` tightens the zero-shot ones by scoring all 2,525. Every adapted
   comparison is reported with that ceiling stated.
10. E14's severity bands were realigned to the annotation rubric before any leaf
   was graded. The two had disagreed by one level — the estimator called anything
   under 5% lesion area `healthy`, while annotators were told under 5% was `mild`
   — which put 85% of the sample in a different bin. Left alone, a flawless
   annotator would have scored κ = 0.63 at 15% exact agreement purely from the
   offset. E12's Dice and E13's ROC-AUC come from continuous ratios and are
   unchanged (0.868 / 0.767 before and after).
