"""Add the architecture figure and the parameter table to the Methods section.

Tutor feedback: the method should list its parameters and be stated alongside a
picture of the model. Only the Models subsection gains content, plus a one-line
edit in EXPERIMENTS so the hyperparameters are not written out twice.
"""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

DOC = Path("report/COMP9444_Project_Report.docx")
INK2 = RGBColor(0x52, 0x51, 0x4E)

LEAD = ("Figure 1 shows how these pieces fit together. A leaf image is augmented, passed through the "
        "three CNNs in parallel, and their softmax outputs are averaged with the validation-tuned "
        "weights to give the 38-way disease prediction; the healthy/diseased answer the brief asks for "
        "is that prediction collapsed to two classes. The severity head is a single regression unit on "
        "the shared ResNet-18 backbone, trained jointly with the classification objective, so severity "
        "is an output of the network rather than a separate pipeline.")

CAPTION = ("Figure 1 — Model architecture. Three CNNs feed a validation-tuned soft-voting ensemble; "
           "the severity head shares the ResNet-18 backbone.")

TABLE_LEAD = ("Table 1 lists every setting. They are held constant across all three architectures and "
              "every experimental arm, so that any difference in results is attributable to the "
              "variable under test rather than to tuning.")

ROWS = [
    ("Input resolution", "128×128 (three baselines) · 224×224 (augmentation, frozen and hierarchical arms)"),
    ("Optimiser", "AdamW"),
    ("Learning rate", "1e-3 · 1e-4 for supervised adaptation on PlantDoc"),
    ("Weight decay", "1e-4"),
    ("Schedule", "Cosine annealing over the epoch budget"),
    ("Loss", "Cross-entropy with 0.1 label smoothing"),
    ("Batch size", "96 · 32 when fine-tuning"),
    ("Epoch budget", "30 baselines · 20 later arms · 15 fine-tuning"),
    ("Early stopping", "Patience 7 on validation accuracy; best checkpoint is evaluated"),
    ("Data split", "70/15/15, grouped by physical leaf, seed 42"),
    ("Ensemble weights", "Grid search in 0.05 steps on the validation split only"),
    ("Severity head", "1 regression unit, jointly trained, loss weight 10"),
]

CAPTION_T = "Table 1 — Training and evaluation parameters, identical across every architecture and arm."


def caption(doc, anchor, text):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = par.add_run(text)
    run.font.size = Pt(8.5)
    run.italic = True
    run.font.color.rgb = INK2
    anchor._p.addnext(par._p)
    return par


def main():
    doc = Document(DOC)
    anchor = next(p for p in doc.paragraphs
                  if p.text.strip().startswith("Severity is estimated from the lesion-area"))
    if any("Figure 1 — Model architecture" in p.text for p in doc.paragraphs):
        raise SystemExit("already added — nothing to do")

    lead = doc.add_paragraph(LEAD)
    anchor._p.addnext(lead._p)

    doc.add_picture("report/figures/architecture.png", width=Inches(6.3))
    pic = doc.paragraphs[-1]
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    lead._p.addnext(pic._p)

    cap = caption(doc, pic, CAPTION)
    tlead = doc.add_paragraph(TABLE_LEAD)
    cap._p.addnext(tlead._p)

    table = doc.add_table(rows=1, cols=2)
    table.style = "Light Grid Accent 1"
    for i, head in enumerate(("Parameter", "Value")):
        run = table.rows[0].cells[i].paragraphs[0].add_run(head)
        run.bold = True
        run.font.size = Pt(9.5)
    for name, value in ROWS:
        cells = table.add_row().cells
        for i, text in enumerate((name, value)):
            run = cells[i].paragraphs[0].add_run(text)
            run.font.size = Pt(9)
    tlead._p.addnext(table._tbl)

    cap_t = doc.add_paragraph()
    cap_t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap_t.add_run(CAPTION_T)
    r.font.size = Pt(8.5); r.italic = True; r.font.color.rgb = INK2
    table._tbl.addnext(cap_t._p)

    # avoid stating the same numbers twice
    for p in doc.paragraphs:
        if p.text.strip().startswith("The settings below were fixed from standard practice"):
            for run in list(p.runs)[1:]:
                run._r.getparent().remove(run._r)
            p.runs[0].text = (
                "The parameters are listed in Table 1 of the Methods section. They were fixed from "
                "standard practice rather than searched, and held constant across every architecture "
                "and arm so that comparisons isolate the variable under test; the ensemble's member "
                "weights are the only component tuned, by grid search on the validation split.")
            break

    doc.save(DOC)
    print("added architecture figure + parameter table to Methods")


if __name__ == "__main__":
    main()
