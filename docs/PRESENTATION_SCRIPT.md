# Presentation script — COMP9444 25T1 Project 090

12 minutes, 5 presenters, 12 slides. Deliver in English; the Chinese lines are
delivery notes, not spoken.

## Slide allocation

The deck currently labels A=2,3 · B=4 · C=5,6 · D=7 · E=8,9,10,11. That gives E
four content slides and B and D one each. The balanced split below is what this
script is written for — **update the `‹Presenter X (zID)›` placeholders to match.**

| | Slides | Time | Content |
|---|---|---|---|
| **A** | 1–3 | 2:20 | Title · motivation · research questions |
| **B** | 4–5 | 2:15 | Literature · datasets |
| **C** | 6–7 | 2:25 | Data analysis · methods |
| **D** | 8–9 | 2:30 | Results I & II |
| **E** | 10–12 | 2:30 | Severity · conclusion · thanks |

The spoken text is **1,270 words**: 10:35 at a careful 120 wpm, 9:45 at a normal
130. Add roughly 40 seconds for four speaker changes and the pauses marked below,
and you land at **11:00–11:30** — inside 12 with margin. 宁可短，不要超时；讲完
留一点时间给提问更好。

If a run-through comes in short, the places to add rather than pad are: on slide
6 name the three dataset variants explicitly, and on slide 9 say which four
probes you mean before listing them.

---

## Presenter A — slides 1–3 (2:20)

### Slide 1 · Title (15s)

> Good morning. We are group 28, and our project is Automatic Plant Disease
> Detection Using Computer Vision — Project 090. I'm ‹name›, and I'll start with
> why this problem is harder than it looks.

### Slide 2 · Motivation (65s)

> Plant disease destroys a large share of global crop yield, and expert
> inspection does not scale. So the brief asks for a CNN that separates healthy
> leaves from diseased ones, to help farmers intervene in time.
>
> We trained on PlantVillage and reached **99.53% across 38 classes**. On most
> projects, that is where the report ends.
>
> We did not stop there, because that phrase — *timely intervention for farmers*
> — is a deployment claim. So we tested the same model on PlantDoc: real
> photographs taken in actual fields. On its 236-image test split, accuracy fell
> from 99.53% to **16.1%**.
>
> That 83-point gap is our project. The rest of this presentation is why it
> happens, and what actually closes it.

*"fell from 99.53% to 16.1%" 是全场最重要的一句，放慢，说完停一秒。第 3/4/5 个方框不用逐字念，手指过去带一句 "those are the three things we did about it" 就行。*

### Slide 3 · Problem statement (62s)

The slide already prints the three questions — so say why each one is there, not
what it is. 这一页最容易变成照着念 PPT；下面没有一句是屏幕上有的。

> Three research questions — and the reason each one is there.
>
> **RQ1** is the brief's task, but we hold ourselves to the harder version.
> Healthy-versus-diseased is a two-way decision; we report all 38 classes,
> because *which* disease is what decides the treatment.
>
> **RQ2** is not in the brief. We added it, because the brief's aim — helping
> farmers intervene — is a deployment claim, and nothing in a laboratory
> benchmark tests deployment.
>
> **RQ3** is harder than it sounds. PlantVillage has no severity labels at all,
> so we had to construct the target ourselves and then prove it means something —
> which is why we validate it against human graders.
>
> The safeguards at the bottom we did not plan. We added them after finding that
> a naive split put photographs of the same physical leaf on both sides. Every
> number you see today is on a split grouped by leaf.
>
> ‹B› will now show that this gap is already well known.

*safeguards 那句照实说"事后才加的"—— 承认踩过坑并回头重建整个项目，比声称全程英明可信得多，tutor 听得出区别。*

---

## Presenter B — slides 4–5 (2:15)

### Slide 4 · Literature (60s)

> The gap we are describing is not new. Mohanty in 2016 reported 99.35% on
> PlantVillage, and noted accuracy dropped sharply under different capture
> conditions. Ferentinos confirmed the same pattern in 2018. Arsenovic attacked
> it in 2019 with heavy augmentation and GAN-generated data. Natarajan added
> explainability in 2024 to see what the model actually looks at.
>
> So the field knows the gap exists. What is done less often is **measuring
> why**. Our contribution is to localise the cause with independent probes, and
> then test whether the standard remedies really work — including honestly
> reporting the ones that do not.

### Slide 5 · Datasets (65s)

> Two datasets. **PlantVillage** is the training corpus: 54,305 photographs of
> single detached leaves, 14 crops, 38 crop-condition classes, shot in a
> laboratory on uniform backgrounds. Every leaf ships in three variants —
> colour, grayscale, and background-removed — and that is what lets us run
> controlled experiments on the background later.
>
> **PlantDoc** is the reality check: 2,525 field photographs, mapped onto 27 of
> the same classes. Whole plants, cluttered backgrounds, overlapping leaves,
> varying scale and lighting. We hold it out completely.
>
> The difference between these two pictures is the entire problem, in one image.

*最后一句配合指屏幕上两组图片。*

---

## Presenter C — slides 6–7 (2:25)

### Slide 6 · Data analysis (75s)

> Three findings from the data shaped everything that follows.
>
> First, the classes are imbalanced about **36 to 1**, so we report macro
> precision, recall and F1 alongside accuracy.
>
> Second — and this is the finding that changed the project — we trained a
> logistic regression on **eight background border pixels**. No leaf at all,
> just the backdrop. It classifies the 38 classes at **33.7%**, against a chance
> rate of 2.6%. That is thirteen times chance, from pixels containing no plant.
> The background carries a large share of the label.
>
> Third, PlantVillage photographs each leaf several times. On a naive random
> split, **74.7%** of our test images had a same-leaf twin sitting in training.
> We rebuilt the entire project on a split grouped by physical leaf — zero
> leakage by construction. Every number you are about to see is on that honest
> split.
>
> Preprocessing is standard: resize, ImageNet normalisation, crop, flip, rotate.
> The non-standard part is background randomisation, which ‹D› will cover.

*这页信息密度最高，三个数字 36:1 / 33.7% / 74.7% 一定要念清楚。*

### Slide 7 · Methods (70s)

> This is the architecture. Three CNNs — a custom network trained from scratch,
> ResNet-18, and MobileNet-V2 — all trained through one identical pipeline, so
> the comparison isolates the architecture. Their softmax outputs are averaged
> in a soft-voting ensemble, with weights grid-searched on the **validation**
> split and applied unchanged to test, so no tuning decision ever sees test data.
>
> The severity head is one regression unit on the shared ResNet-18 backbone,
> trained jointly with the classification objective — so severity is an output
> of the network, not a separate image-processing pipeline.
>
> Every hyperparameter is held constant across all architectures and all
> experimental arms — AdamW, learning rate 1e-3, cosine schedule, label
> smoothing — so any difference in results is attributable to the variable under
> test rather than to tuning.

---

## Presenter D — slides 8–9 (2:30)

### Slide 8 · Results I (70s)

> On the laboratory benchmark, the task is solved. The three baselines land
> between 98.7 and 99.5%, and the ensemble reaches **99.53%** across 38 classes.
>
> Collapsed to the healthy-versus-diseased decision the brief actually asks for,
> it is **99.96%**. Not 100% — of the ensemble's 39 errors, three cross that
> boundary, and two of those are diseased leaves called healthy, which is the
> expensive direction for a farmer.
>
> The confusion matrix is essentially clean. The errors that remain are
> within-crop look-alikes — tomato early blight against late blight — which is
> what a human expert finds hard too.
>
> If we stopped here, this would be a finished project with a 99.5% headline.
> Everything after this slide is why we did not stop.

### Slide 9 · Results II (80s)

> Four probes, each removing one comfort of the benchmark.
>
> Take the **same leaves** and mask out only the background: accuracy drops
> about thirty points. Same leaves, same disease — only the backdrop is gone.
>
> **Grad-CAM**: only 66% of the model's attention falls inside the leaf, while
> the leaf occupies about 50% of the image. The lift over simply staring at the
> middle of the frame is just sixteen points.
>
> On **real field photographs**: 16.1%.
>
> And we tried to patch it at inference — flip-based test-time augmentation, and
> recomputing batch-norm statistics on the target domain. **Both made it
> worse.** That is the informative part. AdaBN repairs an ordinary distribution
> shift; it fails here because this is not a statistics mismatch, it is a
> *learned shortcut*. It cannot be fixed at inference — it has to be fixed in
> training.
>
> More training does not help either. What helps is field data: with only 100
> labelled field images per class, accuracy recovers to 55.5%.

*"Both made it worse" 要停顿 — 这是我们区别于其他组的地方：负结果照实报。*

---

## Presenter E — slides 10–12 (2:30)

### Slide 10 · Severity (80s)

> RQ3 — severity. PlantVillage has no severity labels, so we derive them:
> segment the leaf, count the pixels outside the healthy-green hue band, and
> take the lesion-area fraction.
>
> Two validations. The lesion ratio separates healthy from diseased at **AUC
> 0.87** — a real image feature, no labels needed. And three of us independently
> graded 150 leaves. Our own mutual agreement is only **kappa 0.63** — that is
> the human ceiling; severity is genuinely subjective. The classical estimator
> reaches 0.47 after recalibration, about 74% of that ceiling.
>
> Then we expanded the model, as the brief asks. A severity head *probed* on a
> frozen classification backbone manages only 0.38 within-class correlation —
> classification training discards the colour and texture detail severity needs.
> But trained **jointly**, the same head reproduces the official masks at
> **0.957** within-class, at no cost to classification — 99.44%.
>
> So severity is a genuine model output, from a single forward pass, with no
> mask required at inference.

### Slide 11 · Conclusion (60s)

> To conclude. Both tasks in the brief are delivered: classification at 99.53%,
> healthy-versus-diseased at 99.96%, and severity as a model output.
>
> But the honest headline is the three numbers on this slide. **99.53%** in the
> laboratory. **16.1%** in the field. And **55.9%** once the model actually sees
> field data.
>
> Our recommendation is simple: a benchmark number on PlantVillage is not
> evidence of field readiness, and it should never be reported alone. Report
> field accuracy next to it.
>
> Future work: train on in-field imagery, replace the hue-band lesion rule with
> a learned segmentation model, and calibrate the severity thresholds against
> expert agronomists.

### Slide 12 · Thanks (10s)

> Thank you. The full notebook, all experiment logs and the code are in the
> repository. We are happy to take questions.

---

## Q&A preparation

准备好这几个 — tutor 最可能问的：

**Why is your field accuracy so much lower than published PlantDoc numbers?**
> Because ours is strictly zero-shot: the model never sees a single PlantDoc
> image during training, and we score all 38 classes rather than the 27
> reachable ones. Published numbers usually fine-tune on PlantDoc. When we do
> fine-tune, we get 55.9%, which is in line with the literature.

**Isn't the segmented evaluation unfair — the model never saw a black background?**
> That is exactly the right objection, and it is why we also trained a model
> **on** the segmented variant as the clean control. It reaches 98.66%, so the
> drop is not simply a new domain shift — the shortcut is real.

**Did you tune anything on the test set?**
> No. The ensemble weights are the only tuned component, grid-searched on the
> validation split in 0.05 steps and applied unchanged to test.

**Why McNemar's test instead of confidence intervals?**
> Because the arms are scored on the same images, so the comparisons are paired.
> Overlapping independent intervals would be the wrong test. Single accuracies
> do carry Wilson intervals.

**How do you know the leaf grouping caught all the duplicates?**
> The groups come from `leaf-map.json`, shipped with the dataset. We re-checked
> perceptually with a frozen ImageNet embedding calibrated against
> same-class-different-leaf pairs. The decisive check is that removing the known
> leakage cost only about 0.3 accuracy points — structural leakage would have
> cost far more.

**Did anything you tried fail?**
> Several things, and they are all in the report. A hierarchical crop-then-
> disease head was predicted to lift crop accuracy; it moved it by 0.8 points,
> p = 0.27, and made the healthy/diseased decision significantly worse — we
> record the hypothesis as rejected. Test-time augmentation and AdaBN both hurt.
> And stronger augmentation bought no robustness to capture quality.

---

## Deck changes already applied

In `COMP9444_Presentation_XD_fixed.pptx` — slides 2 and 3 only, nothing else
touched:

- **Slide 2, steps 3–5** stated the safeguards slide 3 lists and the 55.9% slide
  11 concludes with. They now state intent instead, so the same three facts are
  not delivered three times.
- **Slide 3, goal box** was a reworded copy of slide 2's core question. It now
  states the brief's two tasks — new information, not a paraphrase.
- **Slide 2 speaker note** said the model memorises 灰度背景. It does not: the
  shortcut is the **backdrop**, measured by the border-pixel probe at 33.7%
  against 2.6% chance. Grayscale is a separate arm (97.76% trained on it). The
  note is rewritten with the correct mechanism.

## Still to fix

1. **Presenter labels.** All twelve slides still say `‹Presenter A (zID)›`.
   Update them to the balanced split in the table above.
2. **Title slide.** `‹Austral›` is still a placeholder, and only four names are
   listed — the fifth presenter is missing.
3. **Slide 5** reads "4 samples each: colour · grayscale · segmented" — three
   variants under the label "4 samples". Reword to "3 variants per leaf".
4. **Slide 10** has no slide-number box; every other slide has one.
