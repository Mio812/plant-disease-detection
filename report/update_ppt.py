"""Update the UNSW-template PPT body text to honest numbers + the project's thesis.
Preserves layout/formatting: rewrites the run text of each existing bullet in place.
Run: uv run --with python-pptx python report/update_ppt.py
"""
from pptx import Presentation

PPTX = "report/COMP9444_Presentation.pptx"

# slide index (0-based) -> 4 honest bullets replacing the body shape's 4 paragraphs
BULLETS = {
    6: [  # Method(s)
        "Three CNN baselines — Custom CNN (from scratch), ResNet-18, MobileNet-V2 (ImageNet) — combined by a validation-tuned soft-voting ensemble.",
        "Audits of what the model actually uses: background-only probe, Grad-CAM, background randomisation, and a frozen-backbone control.",
        "External validation on held-out PlantDoc field photos, with supervised adaptation from 5 to all shots per class.",
        "Severity: lesion-area ratio from HSV / official masks, plus a regression head trained jointly with the classifier.",
    ],
    5: [  # Data Analysis
        "Strong class imbalance (~36:1) → macro-averaged precision / recall / F1; 72% diseased, 28% healthy.",
        "70/15/15 split by physical leaf (leaf-map.json), so no leaf is shared train ↔ test.",
        "This matters: a naive random split leaks 74.7% of the test set as same-leaf duplicates.",
        "Preprocess: resize + ImageNet normalisation; augment with crop, flip, ±20° rotation, colour jitter.",
    ],
    7: [  # Results
        "Baselines exceed 98.6% on the 8,215-image leaf-grouped test split.",
        "Ensemble (ours) is best: 99.53% over 38 diseases, 99.96% healthy-vs-diseased; 39 errors, none crossing that boundary.",
        "But zero-shot on real PlantDoc field photos collapses to 16.1% — the lab number does not transfer.",
        "Bottleneck is crop identification (39.6%), not diagnosis; adaptation to field data recovers accuracy to 55.9%.",
    ],
    8: [  # Discussion
        "The 99.5% is inflated: 8 background pixels alone classify at 33.7% (chance 2.6%), plus leaf-level train/test leakage.",
        "574× the trainable parameters (full fine-tune vs a frozen 19k-param head) buys 8 lab points and zero field gain.",
        "Severity validated by 3 annotators (κ 0.63 ceiling); trained jointly it is a model output (within-class ρ 0.957), best-in-class but capped near human reliability.",
        "Remaining errors are benign within-crop look-alikes; the model never confuses healthy ↔ diseased.",
    ],
    9: [  # Conclusion
        "Ensemble classifies at 99.53% (38-way) and 99.96% (healthy vs diseased); severity is a jointly-trained model output.",
        "But the lab accuracy is largely capture bias and leaf leakage — shown with controls, not asserted.",
        "It collapses to 16% in the field; the honest deployment path is domain adaptation (→ 55.9%).",
        "Reproducible, config-driven codebase; reporting the field number beside the lab one serves the farmer.",
    ],
}

p = Presentation(PPTX)
for idx, bullets in BULLETS.items():
    body = [sh for sh in p.slides[idx].shapes
            if sh.has_text_frame and sh.text_frame.text.strip()][1]
    paras = body.text_frame.paragraphs
    assert len(paras) == len(bullets), f"slide {idx+1}: {len(paras)} paras vs {len(bullets)} bullets"
    for para, text in zip(paras, bullets):
        para.runs[0].text = text           # keep the run's font, the paragraph's bullet level
        for extra in para.runs[1:]:
            extra._r.getparent().remove(extra._r)

p.save(PPTX)
print(f"updated {len(BULLETS)} slides in {PPTX}")
