"""Fixes for slides 2 and 3 of the presentation deck.

Round 1 -- the slide 2 / slide 3 conflict. Slide 2 was doing three jobs at once:
hook, method, and final result. Steps 3-5 named the exact safeguards slide 3
lists and the exact number slide 11 concludes with, so the deck said the same
things three times and answered RQ1 and RQ2 before slide 3 asked them. Steps 3-5
become forward-looking intent; the specifics stay where they belong. Slide 3's
goal box restated slide 2's core question in different words, so the audience got
two theses back to back; it now states the brief's two tasks instead.

Round 2 -- two defects found on review:
  * 16.1% is the 236-image PlantDoc test split, but slide 5 tells the audience
    PlantDoc is 2,525 images, where the ensemble scores 13.94%. The report
    already writes "16.1% (236-image test split)"; the slide now matches.
  * "held out for external validation" appeared in RQ2's body and again in the
    safeguards row of the same slide.

Assigns run.text, never text_frame.text -- the latter collapses the paragraph to
one unstyled run and loses the template's fonts and colours.
"""
from pathlib import Path

from pptx import Presentation

SRC = Path("deck.pptx")
DST = Path("COMP9444_Presentation_XD_fixed.pptx")

# (slide number, shape id) -> replacement text
EDITS = {
    # slide 2, steps 3-5: drop the method detail (slide 3 + slide 6 own it) and
    # the 55.9% (slide 11 owns it); state the intent instead
    (2, 23): "Find what the model is\nreally using, with\nindependent probes.",
    (2, 29): "Retrain to remove the\nshortcut, and test\nwhether it helps.",
    (2, 35): "Give the model real\nfield data and measure\nwhat returns.",
    # slide 3, goal box: no longer a paraphrase of slide 2's core question
    (3, 7): ("Brief: classify a leaf as healthy or diseased, and estimate disease "
             "severity — plus the field test its stated aim implies."),
    # slide 2, step 2: state the basis -- slide 5 says PlantDoc is 2,525 images,
    # and on all 2,525 this number is 13.94%, not 16.1%
    (2, 17): "Zero-shot on PlantDoc\nfield photos: 16.1%\n(236-image test split).",
    # slide 3, RQ2: the safeguards row below already says PlantDoc is held out
    (3, 19): ("PlantVillage: detached leaves on uniform backgrounds.\n"
              "PlantDoc: whole plants, clutter, and variable scale and lighting."),
}

# the note said the model memorises the grayscale background; the probe measured
# the backdrop, and grayscale is a separate arm
NOTE_2 = (
    "开始用 PlantVillage 训练，发现 acc 过高"
    "（38 类 99.53%）。该数据集拍于实验"
    "室，于是用 PlantDoc 田间照片做零样"
    "本测试，只有 16.1%（236 张测试集；"
    "全部 2,525 张是 13.94%）。\n\n"
    "反推原因：用 8 个背景边缘像素"
    "训练逻辑回归，38 类能到 33.7%（随"
    "机只有 2.6%）—— 背景本身就携带"
    "大量标签信息，模型学到的是拍"
    "摄背景这个捷径，不是病斑。\n"
    "（注意：不是“灰度图”—— "
    "grayscale 是另一个独立实验，在它上"
    "训练得 97.76%。）\n\n"
    "据此做叶片分组切分、背景随机"
    "化和更强增强，最后用田间数据"
    "微调，恢复到 55.9%。"
)


def main():
    prs = Presentation(SRC)
    done = set()
    for (slide_no, shape_id), text in EDITS.items():
        for shape in prs.slides[slide_no - 1].shapes:
            if shape.shape_id != shape_id:
                continue
            runs = [r for p in shape.text_frame.paragraphs for r in p.runs]
            if len(runs) != 1:
                raise SystemExit(f"slide {slide_no} id {shape_id}: expected 1 run, got {len(runs)}")
            print(f"  slide {slide_no} id {shape_id}: {runs[0].text[:45]!r}\n"
                  f"      -> {text[:45]!r}")
            runs[0].text = text
            done.add((slide_no, shape_id))
    missing = set(EDITS) - done
    if missing:
        raise SystemExit(f"shapes not found: {missing}")

    prs.slides[1].notes_slide.notes_text_frame.text = NOTE_2
    print("  slide 2 speaker note rewritten (background shortcut, not grayscale)")

    prs.save(DST)
    print(f"\nwrote {DST}")


if __name__ == "__main__":
    main()
