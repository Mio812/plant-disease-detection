"""Fill the EXPERIMENTS placeholder in the teammate's report with the real setup.

Only that section is touched: the augmentation recipes and hyperparameters the
team asked for, plus the dataset/URL/evaluation items the template prompts for.
Everything else in the document is left exactly as received.
"""
import sys
from pathlib import Path

from docx import Document

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "report/_teammate_version.docx")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "report/COMP9444_Project_Report.docx")

BODY = [
    ("p", "The training corpus is PlantVillage (https://github.com/spMohanty/PlantVillage-Dataset): "
          "54,305 laboratory photographs of single detached leaves, covering 14 crop species and 38 "
          "crop/condition classes (26 diseases plus healthy states). Every image is supplied in three "
          "variants of the same physical leaf — colour, grayscale, and segmented (background removed) — "
          "and we train on all three. External validation uses PlantDoc "
          "(https://github.com/pratikkayal/PlantDoc-Dataset): 2,525 usable in-the-wild field photographs "
          "mapped onto 27 of the same classes, held out entirely and never trained on in any zero-shot "
          "experiment."),
    ("p", "Three findings from the data exploration shaped the design. The classes are imbalanced by "
          "roughly 36:1 (152 images for Potato healthy against 5,507 for Orange citrus greening), so we "
          "report macro-averaged precision, recall and F1 alongside accuracy. Only 28% of the corpus is "
          "healthy leaves. And, most importantly, a logistic regression trained on eight background "
          "border pixels alone — no leaf at all — classifies at 33.7% against a 2.6% chance baseline, "
          "so the backdrop carries a large share of the label."),
    ("h", "Evaluation protocol"),
    ("p", "All models share a single seeded 70/15/15 split (seed 42) computed once and reused across "
          "every model, every dataset variant and every audit. The split is grouped by physical leaf: "
          "PlantVillage photographs each leaf several times, and a naive random split put a same-leaf "
          "twin of 74.7% of test images into training, so the partition is made over leaves rather than "
          "images. Single accuracies carry Wilson 95% intervals; two arms scored on the same images are "
          "compared with McNemar's test on the discordant pairs rather than by asking whether "
          "independent intervals overlap. The ensemble's member weights are the only tuned component: "
          "they are grid-searched in 0.05 steps on the validation split and applied unchanged to test, "
          "so no tuning decision ever sees test data."),
    ("h", "Data augmentation"),
    ("p", "Two recipes are used, and the difference between them matters for the results that follow. "
          "The standard recipe applies a random resized crop retaining 80–100% of the image area, a "
          "random horizontal flip, rotation within ±20°, and colour jitter of ±0.2 in brightness, "
          "contrast and saturation."),
    ("p", "The strong recipe is deliberately more aggressive: a random resized crop down to 50% of the "
          "image area, horizontal flipping plus vertical flipping with probability 0.2, rotation within "
          "±30°, RandAugment with two operations at magnitude 7, colour jitter of ±0.4 in brightness, "
          "contrast and saturation with ±0.1 in hue, Gaussian blur with σ between 0.1 and 1.5, and "
          "random erasing of 2–15% of the image with probability 0.25. Both recipes finish with "
          "ImageNet mean/standard-deviation normalisation."),
    ("p", "A third, targeted intervention is background randomisation. Using the official segmentation "
          "masks, the leaf is cut out and composited onto a randomly generated background — a solid "
          "colour, a linear gradient, uniform noise or blurred blobs — with probability p per image "
          "(p = 0.7 in the reported arms). This exists to break the background/label correlation that "
          "the border-pixel probe exposes, and p = 0.0 with the same strong recipe is kept as the "
          "control so that stronger augmentation and de-shortcutting can be told apart."),
    ("h", "Hyperparameters"),
    ("p", "The settings below were fixed from standard practice rather than searched, and held constant "
          "across every architecture and arm so that comparisons isolate the variable under test. "
          "Optimisation uses AdamW at a learning rate of 1e-3 with weight decay 1e-4 and a cosine "
          "annealing schedule; the loss is cross-entropy with 0.1 label smoothing. Batch size is 96. "
          "Training runs to a maximum of 30 epochs for the baselines and 20 for the later arms, with "
          "early stopping on validation accuracy after 7 epochs without improvement, and the checkpoint "
          "with the best validation accuracy is the one evaluated. Input resolution is 128×128 for the "
          "three baselines and 224×224 for the augmentation, frozen-backbone and hierarchical arms. "
          "Supervised adaptation on PlantDoc uses a lower learning rate of 1e-4, batch size 32 and 15 "
          "epochs, since it starts from an already-trained checkpoint. Every run is seeded at 42."),
]


def main():
    doc = Document(SRC)
    target = next((p for p in doc.paragraphs
                   if p.text.strip().startswith("< Describe briefly about the dataset")), None)
    if target is None:
        raise SystemExit("EXPERIMENTS placeholder not found — has the section already been filled?")

    # reuse the placeholder for the first paragraph, then insert the rest before it
    kind, text = BODY[0]
    for run in list(target.runs)[1:]:
        run._r.getparent().remove(run._r)
    if target.runs:
        target.runs[0].text = text
        target.runs[0].italic = False
    else:
        target.add_run(text)

    # append each block, then move it directly after the previous one
    prev = target
    for kind, text in BODY[1:]:
        new = doc.add_paragraph(text, style="Heading 2" if kind == "h" else "Normal")
        prev._p.addnext(new._p)
        prev = new

    doc.save(DST)
    print(f"filled EXPERIMENTS -> {DST}")


if __name__ == "__main__":
    main()
