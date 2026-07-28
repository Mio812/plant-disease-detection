"""COMP9444 Project 090 report. Honest leaf-grouped rebuild numbers.
Run: uv run --with python-docx python report/build_report.py
"""
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

ACCENT = RGBColor(0x1F, 0x5C, 0x3B)
doc = Document()

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(10.5)
for lvl, sz in ((1, 15), (2, 12.5)):
    st = doc.styles[f"Heading {lvl}"]
    st.font.color.rgb = ACCENT
    st.font.size = Pt(sz)
    st.font.bold = True


def h(text, level=1):
    doc.add_heading(text, level=level)


def p(text, size=None, italic=False, color=None):
    par = doc.add_paragraph()
    run = par.add_run(text)
    if size:
        run.font.size = Pt(size)
    run.italic = italic
    if color:
        run.font.color.rgb = color
    return par


def figure(name, caption):
    doc.add_picture(f"report/figures/{name}", width=Inches(6.1))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = par.add_run(caption)
    run.font.size = Pt(8.5)
    run.italic = True
    run.font.color.rgb = RGBColor(0x52, 0x51, 0x4E)


def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, hdr in enumerate(headers):
        c = t.rows[0].cells[i].paragraphs[0].add_run(hdr)
        c.bold = True
        c.font.size = Pt(9.5)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(9.5)
    doc.add_paragraph()


# ---------------------------------------------------------------- title
title = doc.add_paragraph()
tr = title.add_run("Automatic Plant Disease Detection Using Computer Vision")
tr.bold = True
tr.font.size = Pt(20)
tr.font.color.rgb = ACCENT
sub = doc.add_paragraph()
sr = sub.add_run("COMP9444 Neural Networks and Deep Learning — 25T1 — Project 090")
sr.font.size = Pt(12)
sr.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
p("Group report", size=10, italic=True, color=RGBColor(0x77, 0x77, 0x77))

h("Abstract")
p("The project brief asks for a convolutional neural network that classifies plant leaves as healthy "
  "or diseased and estimates disease severity, in order to aid farmers in timely intervention. We show "
  "that this stated purpose is a deployment claim, and that a benchmark score alone cannot test it. Our "
  "ensemble reaches 99.53% on PlantVillage's 38 specific-disease classes and 99.96% on the healthy-"
  "versus-diseased task. We then demonstrate that this number is inflated by two independent mechanisms "
  "— background capture bias and train/test leakage of duplicate leaves — and that it collapses to 16.1% "
  "on real field photographs, with the bottleneck being crop identification rather than disease diagnosis. "
  "Severity is estimated from image features and, when the backbone is trained jointly for classification "
  "and severity, becomes a genuine model output that reproduces official lesion masks almost perfectly "
  "(within-class rho 0.957) and beats classical estimators on human agreement, though all lesion-area "
  "methods plateau below human reliability. Supervised adaptation to field data recovers accuracy from "
  "16.1% to 55.9%. The honest deployment story is domain adaptation, not a headline accuracy.")

h("1. Introduction")
p("Plant diseases reduce agricultural productivity and threaten food security. Expert visual inspection "
  "is slow, subjective and does not scale, so the brief asks for a CNN that classifies leaves and "
  "estimates severity, “to aid farmers and agricultural experts in timely intervention.” That "
  "closing phrase is a deployment claim, and it shapes the whole study. The PlantVillage dataset the "
  "brief points to is 54,305 studio photographs: single detached leaves, flat, centred, on uniform "
  "backgrounds under even lighting. Any modern CNN exceeds 99% on it, but that number cannot answer the "
  "brief's own question, because a farmer photographs whole plants in a field, not detached leaves in a studio.")
p("We therefore split the work into three research questions:")
table(["", "Research question", "Brief"],
      [["RQ1", "Can a CNN classify PlantVillage leaves accurately?", "Task 1"],
       ["RQ2", "Does that accuracy mean the system works on real field photographs?", "implied by the aim"],
       ["RQ3", "Can severity be estimated from image features, and is it trustworthy?", "Task 2"]])
p("RQ1 turns out to be easy, and that is precisely why RQ2 matters. Much of the report is the discipline "
  "of not taking 99% at face value: measuring what the classifier actually uses, auditing the integrity "
  "of the test set itself, and validating on genuine field imagery from the PlantDoc dataset, which is "
  "never used in training.")

h("2. Related work")
p("Mohanty et al. (2016) established the PlantVillage benchmark and the near-ceiling accuracies that "
  "follow from it. Ferentinos (2018) scaled CNN plant-disease classification across architectures and "
  "noted the gap between laboratory and real conditions. Singh et al. (2020) introduced PlantDoc precisely "
  "to expose that gap: a curated set of field photographs on which PlantVillage-trained models transfer "
  "poorly, and showed that adding in-domain data improves accuracy by up to 31 percentage points. Our work "
  "reproduces the collapse, decomposes its cause, and audits an internal validity threat — leaf-level "
  "train/test leakage — that inflates the PlantVillage number independently of the domain gap.")

h("3. Dataset and methods")
h("3.1 Data", 2)
p("PlantVillage provides 54,305 images across 38 classes (14 crop species, 26 diseases, plus healthy "
  "states) in three variants — colour, grayscale, and segmented (background removed) — of the same "
  "physical leaves. We use all three. A single seeded split (70/15/15) is shared by every model and "
  "variant. PlantDoc supplies 2,525 usable field photographs mapped to 27 of the classes; it is held out "
  "entirely and used only for external validation.")
p("A subtle but material property of PlantVillage is that each physical leaf is photographed several "
  "times. The repository ships leaf-grouping metadata recording which images share a leaf. A naive random "
  "split scatters these near-duplicates across train and test; we measured that 74.7% of a random test "
  "split had a same-leaf twin in training. We therefore partition by leaf within each class, so no leaf's "
  "images straddle the split. This is the split used throughout.")
h("3.2 Models", 2)
p("Three CNNs span two families and a range of sizes: a four-block CNN trained from scratch, ImageNet-"
  "pretrained ResNet-18, and MobileNet-V2. All share one pipeline. Our proposed model is their soft-voting "
  "ensemble, with weights chosen by grid search on the validation split and applied unchanged to test. "
  "Severity is estimated from the lesion-area ratio — the fraction of leaf pixels showing symptoms — "
  "computed from either an unsupervised (Otsu) mask or the official segmented mask, and from a severity-"
  "regression head trained jointly with the classifier (Section 7).")

h("4. Results — Task 1: classification")
p("On the held-out PlantVillage test split (8,215 images), the ensemble reaches 99.53% over the 38 "
  "specific-disease classes and 99.96% on the binary healthy-versus-diseased task the brief names.")
table(["Model", "38-way (specific disease)"],
      [["Custom CNN (from scratch)", "98.69%"], ["ResNet-18", "99.33%"],
       ["MobileNet-V2", "99.46%"], ["Ensemble (proposed)", "99.53%"],
       ["Ensemble — binary healthy/diseased", "99.96%"]])
p("The 39 residual errors out of 8,215 are almost all within-crop disease confusions that a specialist "
  "would find genuinely ambiguous (for example Tomato early versus late blight). Three of them do cross "
  "the healthy/diseased boundary — and two of those are diseased leaves called healthy, the costlier "
  "direction for a farmer — so the binary task is near-solved at 99.96% rather than perfect. Across the "
  "dataset's three variants the model holds: colour 99.53%, segmented 98.66%, grayscale 97.76%.")
figure("per_disease_accuracy.png",
       "Figure 1 — Per-disease recall on the leaf-grouped test split. 29 of 38 classes are perfect; "
       "the five in red are within-crop look-alikes.")
figure("confusion_matrix.png",
       "Figure 2 — Ensemble confusion matrix (row-normalised). The diagonal is essentially clean; red "
       "digits are the 39 misclassified images.")

h("5. Results — is the 99% real?")
p("Two independent audits show that the headline number overstates real diagnostic skill.")
h("5.1 Capture bias", 2)
p("A classifier trained on eight background border pixels alone — never seeing the leaf — reaches 33.7% "
  "accuracy, against a 2.6% chance baseline. Grad-CAM attention places 65.6% of its mass inside the leaf, "
  "only modestly above the 49.6% the leaf occupies by area. Both show the model reads the background, "
  "which correlates with the label because each class was photographed in its own setting.")
h("5.2 Leaf-level leakage", 2)
p("Adopting the leaf-grouped split lowers the ensemble from 99.84% to 99.53% — a 0.31-point drop, small "
  "because the classification is genuinely strong, but real. Per disease the effect concentrates: Tomato "
  "early blight falls from a leaked 99% to an honest 93%, exposing the one diagnostic weakness the leaked "
  "test had hidden. A perceptual re-check bounds the residual leakage the metadata missed at under 10.7% "
  "(an over-estimate, since PlantVillage's near-identical distinct leaves inflate it); the 0.31-point "
  "accuracy cost confirms the residual is immaterial.")

h("6. Results — does it survive the field?")
p("Zero-shot on PlantDoc, the ensemble scores 16.1% (236-image test split) and 13.9% across all 2,525 "
  "images — a collapse from 99.5% in the laboratory. This is the answer to RQ2, and it is the central "
  "result: the benchmark accuracy does not transfer to the conditions the brief describes.")
figure("lab_vs_field.png",
       "Figure 3 — The same ensemble on studio and field photographs. Fine-grained accuracy collapses; "
       "even the coarse healthy/diseased decision loses ~20 points.")

h("6.1 Where the accuracy goes", 2)
p("Decomposing field accuracy into crop identification and disease-given-crop shows the bottleneck is "
  "recognising the plant, not the disease: crop accuracy is only 39.6%. Background randomisation during "
  "training significantly improves field crop identification (McNemar p = 0.0013), and a frozen ImageNet "
  "backbone — training only a 19,494-parameter head, 574 times fewer parameters than full fine-tuning — "
  "matches or beats full fine-tuning in the field while costing eight points in the laboratory. The "
  "laboratory points that extra training buys are worth nothing, sometimes less than nothing, outside "
  "the benchmark.")
table(["Arm (field, 2,525 images)", "Trainable params", "Lab", "Field", "Crop"],
      [["Full fine-tune", "11,196,006", "99.20%", "15.2%", "37.2%"],
       ["Frozen backbone", "19,494", "91.26%", "16.9%", "39.4%"],
       ["Frozen + bg-random", "19,494", "88.41%", "18.1%", "41.4%"]])
figure("frozen_vs_full.png",
       "Figure 4 — The strongest control: training 574x fewer parameters costs ~8 points in the "
       "laboratory and loses nothing in the field.")

h("6.2 Robustness and adaptation", 2)
p("Field accuracy is nearly flat across capture-quality bins (luminance, contrast, and sharpness gaps "
  "under 1.6 points), so the failure is not caused by poor photographs — it is uniform across bright, "
  "dark, sharp and blurry images. The domain gap is compositional (whole plants, overlapping foliage, "
  "varying scale), not photometric. The honest remedy is target-domain data: fine-tuning on PlantDoc "
  "raises accuracy from 16.1% through 47.0% at 20 shots per class to 55.9% with all field data, an "
  "improvement of 39.8 points over the zero-shot baseline that meets the reference dataset's own "
  "31-point improvement target.")

figure("adaptation_curve.png",
       "Figure 5 — Supervised adaptation on field data, the honest remedy: 16.1% zero-shot to 55.9% "
       "with the full field training set.")

h("7. Results — Task 2: severity")
p("PlantVillage carries no severity labels, so severity is derived from the lesion-area ratio. As a "
  "presence signal it is strong: ROC-AUC 0.868 with the official mask, 0.767 with unsupervised Otsu "
  "segmentation. Validated against three human annotators who graded 150 leaves, the annotators agree "
  "with each other at quadratic kappa 0.634 (the ceiling); the lesion-ratio grade tracks their consensus "
  "at kappa 0.30 with the original bands, rising to 0.47 after cross-validated recalibration — a moderate "
  "proxy reaching about three-quarters of the human ceiling.")
p("To satisfy the brief's requirement to expand the model, we trained a severity-regression head jointly "
  "with the classifier. This transforms the result: the joint head reproduces the official-mask lesion "
  "ratio almost perfectly (within-class rho 0.957) at no classification cost (99.44%), where a probe on "
  "frozen classification features had managed only 0.38. Severity is now a genuine model output requiring "
  "no mask at inference. On the subset of graded leaves the model never trained on, it agrees with human "
  "consensus at rho 0.47, ahead of both Otsu (0.43) and the official mask (0.41). The head is therefore "
  "the best severity estimator available and a model output — but all lesion-area methods plateau near "
  "rho 0.47 against humans, because people grade severity by more than the fraction of leaf area affected. "
  "We report this as the ceiling of image-feature severity estimation rather than inflate it.")

figure("severity.png",
       "Figure 6 — Severity. Left: the grade against three human annotators, with their mutual "
       "agreement as the ceiling. Right: trained jointly with the classifier, severity becomes a "
       "model output that beats the classical estimator.")

h("8. Discussion")
p("The project's contribution is not a 99% accuracy; it is the demonstration, with controlled experiments "
  "rather than assertion, that this number is largely capture bias and leaf leakage, and that it does not "
  "survive contact with the field the brief describes. Three results carry the argument: 574 times the "
  "trainable parameters buy no field accuracy; removing colour costs one laboratory point but fifteen "
  "field points, showing the benchmark cannot distinguish a model that reads symptoms from one that does "
  "not; and the field error decomposes cleanly into a crop-identification bottleneck that augmentation "
  "cannot fix, because PlantVillage contains no whole-plant, multi-leaf, distant views to learn from.")
p("Limitations. Our field evaluation on the 236-image test split carries a Wilson interval of roughly "
  "five points; we therefore report full-dataset figures alongside it. PlantDoc carries label noise and "
  "its train/test splits differ in content. Severity is validated on 150 leaves by three annotators, and "
  "the human cross-check of the joint head rests on the 20 of those leaves in the model's test split, so "
  "its ordering is indicative rather than conclusive. Growth-stage robustness cannot be tested because "
  "PlantDoc carries no stage labels; we record it as a limitation rather than proxy it.")

h("9. Conclusion")
p("A CNN ensemble classifies PlantVillage leaves at 99.5% and healthy-versus-diseased at essentially "
  "100%, and a jointly trained model estimates severity from image features as a genuine output. But the "
  "brief's deployment aim is met only under honest scrutiny: the laboratory accuracy is inflated by "
  "capture bias and leaf leakage and collapses to 16% in the field, where the binding constraint is crop "
  "identification. Supervised adaptation to field data is the truthful path forward, recovering accuracy "
  "to 56%. Reporting the field number honestly beside the laboratory one — rather than the 99% alone — is "
  "the result most useful to the farmer the brief describes.")

h("References")
for ref in [
    "Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. Computers "
    "and Electronics in Agriculture, 145, 311–318.",
    "Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant "
    "disease detection. Frontiers in Plant Science, 7, 1419.",
    "Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A dataset for "
    "visual plant disease detection. In Proc. 7th ACM IKDD CoDS and 25th COMAD, 249–253.",
]:
    p(ref, size=9)

doc.core_properties.author = "COMP9444 25T1 Project 090"
doc.core_properties.title = "Automatic Plant Disease Detection Using Computer Vision"
doc.save("report/COMP9444_Project_Report.docx")
print("wrote report/COMP9444_Project_Report.docx")
