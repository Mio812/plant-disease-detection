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
| E11 | Few-shot fine-tune on PlantDoc train | supervised ceiling | **reported separately** — touches target labels | open |
| E12 | Leaf segmentation vs official masks (Dice) | H6 | official mask is ground truth | done |
| E13 | Lesion ratio separates healthy vs diseased (ROC-AUC) | H6 | needs no manual labels | done |
| E14 | Ordinal grade vs manual annotation (ρ, MAE, κ) | H6 | 150 leaves, graded by the team | open |

## 4. What would falsify the conclusions

- If E3 scored near 2.6%, and E5 showed attention concentrated on the leaf,
  H2 would be rejected and the 99.8% would be taken at face value.
- If E8 with `p = 0.0` matched `p = 0.7` on PlantDoc, the gain would be
  attributable to augmentation alone and H5 would be rejected.
- If E9 (trained on `segmented`) scored poorly *in-domain*, the E4 collapse
  would be explained by loss of information rather than loss of a shortcut.
- If E13's AUC were near 0.5, the severity signal would be meaningless.

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

## 6. Reporting protocol

1. Lab and field accuracy are always reported **together**; a lab number alone
   is not a result.
2. A drop in PlantVillage accuracy when the shortcut is removed is expected and
   is *not* a regression — it is the price of generalisation.
3. E11 never shares a row with zero-shot numbers.
4. Known threats to validity: PlantDoc is web-scraped and carries label noise;
   its train/test splits are known to differ in content; our field evaluation is
   236 images (±5 points), so the full-dataset variant is also reported.
