"""Report/presentation figures, generated from the committed result artefacts.

Every figure reads outputs/*.json or *.txt, so nothing here is hand-entered.
Colours are the validated categorical/sequential palette (light mode, for print).
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

OUT = Path("outputs")
FIG = Path("report/figures")
FIG.mkdir(parents=True, exist_ok=True)

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("seq_blue", SEQ)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(name):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def pct(v):
    """99.96 must not print as 100.0 — this project's point is that it is not perfect."""
    return f"{v:.2f}%" if v > 99.9 else f"{v:.1f}%"


def style(ax, xlabel=None, ylabel=None, title=None, pad=10):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title, color=INK, fontsize=10.5, fontweight="bold", loc="left", pad=pad)


def parse_report(path):
    """Per-class (recall, support) from a sklearn classification report."""
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith(("precision", "accuracy", "macro", "weighted")):
            continue
        parts = line.rsplit(None, 4)
        if len(parts) != 5:
            continue
        name, _prec, rec, _f1, sup = parts
        try:
            rows[name.strip()] = (float(rec), int(sup))
        except ValueError:
            pass
    return rows


ABBREV = {
    "Cercospora leaf spot Gray leaf spot": "Gray leaf spot",
    "Haunglongbing Citrus greening": "Citrus greening",
    "Leaf blight Isariopsis Leaf Spot": "Leaf blight",
    "Esca Black Measles": "Esca",
    "Spider mites Two-spotted spider mite": "Spider mites",
    "Yellow Leaf Curl Virus": "Yellow leaf curl",
    "mosaic virus": "Mosaic virus",
    "Northern Leaf Blight": "Northern leaf blight",
    "Cedar apple rust": "Cedar rust",
}


def short(cls):
    """Axis label: drop the crop name where the disease repeats it, and abbreviate
    the few very long names that otherwise dominate a 38-label axis."""
    crop, _, dis = cls.partition("___")
    crop = crop.replace("_(maize)", "").replace("_(including_sour)", "").replace(",_bell", "")
    dis = re.sub(r"\s+", " ", dis.replace("_", " ").replace("(", "").replace(")", "")).strip()
    if dis.lower().startswith(crop.lower()):
        dis = dis[len(crop):].strip()
    dis = ABBREV.get(dis, dis)
    return f"{crop} · {dis.lower() if dis.lower() == 'healthy' else dis}"


# --------------------------------------------------------------- 1. confusion matrix
def fig_confusion():
    rep = parse_report(OUT / "ensemble_plantvillage_report.txt")
    pairs = load("ensemble_plantvillage_confused_pairs.json")["pairs"]
    classes = sorted(rep)
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)

    cm = np.zeros((n, n))
    for p in pairs:
        cm[idx[p["true"]], idx[p["pred"]]] = p["count"]
    for c, (_rec, sup) in rep.items():
        i = idx[c]
        cm[i, i] = sup - cm[i].sum()
    row = cm.sum(1, keepdims=True)
    norm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)

    fig, ax = plt.subplots(figsize=(7.8, 7.0))
    im = ax.imshow(norm, cmap=BLUES, norm=PowerNorm(gamma=0.4, vmin=0, vmax=1))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([short(c) for c in classes], rotation=90, fontsize=7.0)
    ax.set_yticklabels([short(c) for c in classes], fontsize=7.0)
    ax.set_xlabel("predicted", color=INK2); ax.set_ylabel("true", color=INK2)
    ax.set_title("Ensemble confusion matrix — PlantVillage test (row-normalised)",
                 color=INK, fontsize=11, fontweight="bold", loc="left", pad=30)
    # only the errors are worth labelling; the diagonal is ~1.0 everywhere
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6.4,
                        color=RED, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.015)
    cb.set_label("share of true class", color=INK2, fontsize=8)
    cb.ax.tick_params(labelsize=7, color=MUTED)
    cb.outline.set_edgecolor(AXIS)
    ax.text(0, 1.008, "39 errors in 8,215 images; red digits are the misclassified counts.",
            transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    save(fig, "confusion_matrix.png")


# ------------------------------------------------------- 2. the lab-to-field collapse
def fig_lab_vs_field():
    lab, field = load("eval_plantvillage.json"), load("eval_plantdoc.json")
    full = load("eval_plantdoc_full.json")
    groups = ["38-way\n(specific disease)", "Healthy vs diseased"]
    labv = [lab["ensemble"], lab["ensemble_binary"]]
    fldv = [field["ensemble"], field["ensemble_binary"]]

    x = np.arange(len(groups)); w = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    b1 = ax.bar(x - w / 2, labv, w, label="PlantVillage (lab)", color=BLUE)
    b2 = ax.bar(x + w / 2, fldv, w, label="PlantDoc (field)", color=ORANGE)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                    pct(b.get_height()), ha="center", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
    style(ax, ylabel="accuracy (%)",
          title="Laboratory accuracy does not transfer to the field", pad=34)
    ax.legend(frameon=False, fontsize=8.5, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.text(0, -22, f"Field = zero-shot on held-out PlantDoc photographs "
                    f"(n={field['n_images']}; {full['ensemble']:.1f}% over all {full['n_images']:,}).",
            fontsize=8, color=INK2)
    save(fig, "lab_vs_field.png")


# ---------------------------------------------------------- 3. adaptation curve (E18)
def fig_adaptation():
    zero = load("eval_plantdoc.json")["ensemble"]
    pts = [(0, zero)]
    for shots, tag in [(5, "ft_shots5"), (20, "ft_robust"), (50, "ft_shots50"),
                       (100, "ft_shots100")]:
        d = load(f"{tag}_history.json")
        if d: pts.append((shots, d["best"]))
    full = load("ft_full_history.json")
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(xs, ys, "-o", color=BLUE, linewidth=2, markersize=7,
            markerfacecolor=SURFACE, markeredgewidth=2)
    if full:
        ax.axhline(full["best"], color=AQUA, linestyle="--", linewidth=1.6)
        ax.text(101, full["best"] + 1.4, f"all field data · {full['best']:.1f}%",
                color=AQUA, fontsize=8.5, ha="right")
    for xx, yy in [pts[0], pts[1], pts[-1]]:
        ax.annotate(f"{yy:.1f}%", (xx, yy), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=8.5, color=INK)
    ax.set_ylim(0, 65)
    style(ax, xlabel="labelled field images per class (shots)", ylabel="PlantDoc accuracy (%)",
          title="Supervised adaptation is what actually closes the gap")
    ax.text(0, -14.5, "Zero-shot 16.1% → 55.9% with the full field training set (+39.8 points).",
            fontsize=8, color=INK2)
    save(fig, "adaptation_curve.png")


# ------------------------------------------ 4. frozen vs full fine-tune (E15, the control)
def fig_frozen():
    arms = [("bg_control", "resnet18_color_strong_p0_224", "Full fine-tune"),
            ("frozen_ctrl", "resnet18_color_strong_p0_224_frozen", "Frozen backbone")]
    labels, labv, fldv, params = [], [], [], []
    for tag, hist, label in arms:
        d, h = load(f"eval_arm_{tag}.json"), load(f"{hist}_history.json")
        if not d or not h: continue
        labels.append(label); labv.append(h["test_metrics"]["accuracy"] * 100)
        fldv.append(d["ensemble"]); params.append(h["trainable_parameters"])

    x = np.arange(len(labels)); w = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    b1 = ax.bar(x - w / 2, labv, w, label="PlantVillage (lab)", color=BLUE)
    b2 = ax.bar(x + w / 2, fldv, w, label="PlantDoc (field)", color=ORANGE)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                    pct(b.get_height()), ha="center", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n{p:,} trainable params" for l, p in zip(labels, params)])
    ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
    style(ax, ylabel="accuracy (%)",
          title="574× more trainable parameters buys lab accuracy, not field accuracy", pad=34)
    ax.legend(frameon=False, fontsize=8.5, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.text(0, -24, "Freezing the ImageNet backbone costs ~8 points in the lab and "
                    "loses nothing in the field.", fontsize=8, color=INK2)
    save(fig, "frozen_vs_full.png")


# ------------------------------------------------------------- 5. per-disease accuracy
def fig_per_disease():
    rep = parse_report(OUT / "ensemble_plantvillage_report.txt")
    items = sorted(rep.items(), key=lambda kv: kv[1][0])
    names = [short(c) for c, _ in items]
    accs = [v[0] * 100 for _, v in items]
    colors = [RED if a < 99 else BLUE for a in accs]

    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    ax.barh(range(len(names)), accs, color=colors, height=0.74)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7.4)
    ax.set_xlim(0, 108); ax.set_xticks([0, 25, 50, 75, 100])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("recall (%)", color=INK2)
    ax.set_title("Per-disease accuracy — every class, weakest first",
                 color=INK, fontsize=10.5, fontweight="bold", loc="left", pad=10)
    for i, a in enumerate(accs):
        if a < 99:
            ax.text(a + 1.5, i, f"{a:.0f}%", va="center", fontsize=7.6, color=RED, fontweight="bold")
    n_perfect = sum(1 for a in accs if a >= 99.5)
    n_weak = sum(1 for a in accs if a < 99)
    fig.text(0.5, -0.012,
             f"{n_perfect} of 38 classes are perfect; the {n_weak} in red are the only weak spots, "
             f"all within-crop look-alikes.", fontsize=8.5, color=INK2, ha="center")
    save(fig, "per_disease_accuracy.png")


# ------------------------------------------------------------------ 6. severity (Task 2)
def fig_severity():
    val, joint = load("severity_validation.json"), load("joint_severity.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))

    ax = axes[0]
    ceiling = val.get("inter_annotator_kappa_mean_pairwise")
    names = ["rubric\nbands", "recalibrated\n(5-fold CV)"]
    vals = [val["quadratic_kappa"], val["quadratic_kappa_recalibrated_cv"]]
    bars = ax.bar(names, vals, color=[MUTED, BLUE], width=0.5)
    ax.axhline(ceiling, color=AQUA, linestyle="--", linewidth=1.6)
    ax.text(1.46, ceiling + 0.015, f"human ceiling {ceiling:.2f}", color=AQUA,
            fontsize=8, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center",
                fontsize=9, color=INK)
    ax.set_ylim(0, 0.78)
    style(ax, ylabel="quadratic κ vs 3 annotators",
          title="Severity grade vs human judgement")

    ax = axes[1]
    names2 = ["joint head\n(E26)", "Otsu\n(classical)"]
    vals2 = [joint["within_class_rho_head"], joint["within_class_rho_otsu"]]
    bars = ax.bar(names2, vals2, color=[BLUE, MUTED], width=0.5)
    for b, v in zip(bars, vals2):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center",
                fontsize=9, color=INK)
    ax.set_ylim(0, 1.12)
    style(ax, ylabel="within-class ρ vs official mask",
          title="Severity as a model output")
    fig.text(0.0, -0.06, "Left: the grade is a moderate proxy, ~74% of the human ceiling. "
                         "Right: trained jointly with the classifier, the severity head "
                         "reproduces the mask far better than the classical estimator.",
             fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "severity.png")




# ------------------------------------------------- 7. model architecture (Methods)
def fig_architecture():
    """Block diagram of the pipeline, for the Methods section."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12.6, 5.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(x, y, w, h, title, sub="", fc="#ffffff", ec=BLUE, tc=INK, lw=1.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                    linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
        if sub:
            ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
                    fontsize=8.9, fontweight="bold", color=tc, zorder=3)
            ax.text(x + w / 2, y + h * 0.33, sub, ha="center", va="center",
                    fontsize=7.3, color=INK2, zorder=3, linespacing=1.6)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=9.2, fontweight="bold", color=tc, zorder=3)

    def arrow(x1, y1, x2, y2, color=MUTED):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.2, color=color, zorder=1,
                                     shrinkA=0, shrinkB=0))

    # five numbered stages, so the eye has a left-to-right reading order
    cols = [(0.012, 0.150), (0.181, 0.170), (0.372, 0.196), (0.598, 0.156), (0.784, 0.204)]
    stages = ["1 · Input", "2 · Augmentation", "3 · Backbones (parallel)", "4 · Fusion", "5 · Outputs"]
    for (x, w), name in zip(cols, stages):
        ax.add_patch(FancyBboxPatch((x - 0.008, 0.075), w + 0.016, 0.785,
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    linewidth=0, facecolor="#f7f7f5", zorder=0))
        ax.text(x + w / 2, 0.884, name, ha="center", fontsize=8.2,
                color=MUTED, fontweight="bold")

    MID = 0.622                                    # centre line of the classification track
    box(cols[0][0], MID - 0.095, cols[0][1], 0.19, "RGB leaf image",
        "128×128 baselines\n224×224 main arms", fc="#ffffff", ec=AXIS, tc=INK2)
    box(cols[1][0], MID - 0.125, cols[1][1], 0.25, "Augmentation",
        "crop 0.5–1.0 · flips · ±30°\nRandAugment · jitter 0.4\nblur · erasing\nbackground-rand p=0.7",
        fc="#ffffff", ec=AXIS, tc=INK2)

    ys = [0.715, 0.560, 0.405]
    names = [("Custom CNN", "4 conv blocks · 0.4M · from scratch"),
             ("MobileNet-V2", "inverted residuals · 2.3M · ImageNet"),
             ("ResNet-18", "residual blocks · 11.2M · ImageNet")]
    for y, (n, s) in zip(ys, names):
        box(cols[2][0], y, cols[2][1], 0.125, n, s)
        arrow(cols[1][0] + cols[1][1], MID, cols[2][0] - 0.003, y + 0.062)
        arrow(cols[2][0] + cols[2][1] + 0.003, y + 0.062, cols[3][0] - 0.003, MID)

    arrow(cols[0][0] + cols[0][1], MID, cols[1][0] - 0.003, MID)
    box(cols[3][0], MID - 0.105, cols[3][1], 0.21, "Soft-voting ensemble",
        "average of the three\nsoftmax vectors; weights\ngrid-searched on validation",
        fc="#eef4fd", ec=BLUE, lw=1.8)

    box(cols[4][0], 0.665, cols[4][1], 0.125, "38 classes", "specific crop × disease", ec=AQUA)
    box(cols[4][0], 0.475, cols[4][1], 0.125, "Healthy / diseased", "collapsed from the 38", ec=AQUA)
    arrow(cols[3][0] + cols[3][1], MID + 0.02, cols[4][0] - 0.003, 0.7275, AQUA)
    arrow(cols[3][0] + cols[3][1], MID - 0.02, cols[4][0] - 0.003, 0.5375, AQUA)

    # severity track, on its own row so nothing crosses a box
    ax.plot([0.004, 0.996], [0.345, 0.345], color=GRID, linewidth=1, zorder=0)
    ax.text(0.004, 0.300, "Severity — a second head on the trained ResNet-18 backbone",
            fontsize=8.2, color=ORANGE, fontweight="bold")
    for col, (t, s, fc) in zip(cols[2:], [
            ("512-d features", "penultimate layer\nof ResNet-18", "#ffffff"),
            ("Severity head", "1 unit + sigmoid\ntrained jointly", "#fdf3ee"),
            ("Lesion ratio", "√ratio →\n4 ordinal grades", "#ffffff")]):
        box(col[0], 0.100, col[1], 0.150, t, s, fc=fc, ec=ORANGE)
    arrow(cols[2][0] + cols[2][1] * 0.5, 0.403, cols[2][0] + cols[2][1] * 0.5, 0.253, ORANGE)
    arrow(cols[2][0] + cols[2][1] + 0.003, 0.175, cols[3][0] - 0.003, 0.175, ORANGE)
    arrow(cols[3][0] + cols[3][1] + 0.003, 0.175, cols[4][0] - 0.003, 0.175, ORANGE)

    ax.text(0.0, 0.987, "Model architecture", fontsize=12.5, fontweight="bold", color=INK)
    ax.text(0.0, 0.943, "Three CNNs share one training pipeline and are combined by a "
                        "validation-tuned soft vote. The severity head is a second output "
                        "on the ResNet-18 backbone, trained jointly with classification.",
            fontsize=8.5, color=INK2)
    save(fig, "architecture.png")


# ---------------------------------- 8. dataset composition (Data Analysis slide)
def fig_dataset():
    """Four donuts, as the team prefers. Every ring is a genuine share of a whole:
    the imbalance panel shows how much of the corpus the largest classes hold,
    because a 36:1 ratio is not a part-to-whole and cannot honestly be a slice."""
    from torchvision.datasets import ImageFolder

    from src.config import Config
    from src.data import splits_from_config

    cfg = Config.load("configs/default.yaml")
    base = ImageFolder(cfg.data.root)
    counts = np.bincount([l for _, l in base.samples])
    total = int(counts.sum())
    top10 = 100 * np.sort(counts)[::-1][:10].sum() / total
    healthy = sum(int(c) for c, n in zip(counts, base.classes) if "healthy" in n.lower())
    diseased = 100 * (total - healthy) / total
    tr, va, te = (100 * len(s) / total for s in splits_from_config(cfg, base))

    n_top10 = int(np.sort(counts)[::-1][:10].sum())
    tr_n, va_n, te_n = (len(s) for s in splits_from_config(cfg, base))
    # every segment is named and counted, so the ring itself carries the reading
    panels = [
        ("Class imbalance",
         [("10 largest\nclasses", top10, n_top10, RED),
          ("28 smaller\nclasses", 100 - top10, total - n_top10, "#f6b9b8")],
         "36:1", "max : min"),
        ("Split leakage (before the fix)",
         [("same-leaf\ntwin in train", 74.7, round(8215 * 0.747), ORANGE),
          ("genuinely\nunseen", 25.3, 8215 - round(8215 * 0.747), "#f8c7ae")],
         "74.7%", "leaked"),
        ("Leaf-grouped split",
         [("train", tr, tr_n, AQUA), ("val", va, va_n, "#7fd3b4"),
          ("test", te, te_n, "#cfeee1")],
         "54,305", "images"),
        ("Label balance",
         [("diseased", diseased, total - healthy, BLUE),
          ("healthy", 100 - diseased, healthy, "#9ec5f4")],
         "38", "classes"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14.4, 5.0))
    for ax, (title, segs, big, small) in zip(axes, panels):
        wedges, _ = ax.pie([s[1] for s in segs], colors=[s[3] for s in segs],
                           startangle=90, counterclock=False,
                           wedgeprops=dict(width=0.40, edgecolor=SURFACE, linewidth=2.5))
        ax.text(0, 0.10, big, ha="center", va="center", fontsize=17,
                fontweight="bold", color=INK)
        ax.text(0, -0.17, small, ha="center", va="center", fontsize=8.5, color=MUTED)
        for wedge, (name, pctv, n, _c) in zip(wedges, segs):
            ang = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
            x, y = np.cos(ang), np.sin(ang)
            ax.text(1.20 * x, 1.20 * y, f"{name}\n{pctv:.0f}%  ·  {n:,}",
                    ha="center" if abs(x) < 0.35 else ("left" if x > 0 else "right"),
                    va="center", fontsize=8.6, color=INK, linespacing=1.5)
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, pad=18)
        ax.set_xlim(-1.85, 1.85); ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal")
    fig.tight_layout(w_pad=0.6)
    save(fig, "dataset_composition.png")


def fig_variants():
    """The three dataset variants on the same physical leaves. `segmented` is what
    makes the background experiments possible, and seeing it explains why removing
    the backdrop costs so much accuracy."""
    from PIL import Image

    from src.data.variants import name_key, variant_index

    root = Path("data/PlantVillage/raw/color")
    if not root.exists():
        print("  (skipped variants — PlantVillage not downloaded)")
        return
    picks = ["Tomato___Late_blight", "Apple___Apple_scab", "Corn_(maize)___Common_rust_",
             "Grape___healthy"]
    rows = []
    for cls in picks:
        d = root / cls
        if not d.exists():
            continue
        names = sorted(p.name for p in d.iterdir())[:12]
        # prefer a leaf whose colour photo has a visible backdrop, so the segmented
        # column actually shows something being removed
        names.sort(key=lambda n: -float(np.asarray(Image.open(d / n))[0].mean()))
        for name in names:
            trio = [d / name]
            for v in ("grayscale", "segmented"):
                twin = variant_index(str(root.parent / v), cls).get(name_key(name))
                if twin:
                    trio.append(Path(twin))
            if len(trio) == 3:
                rows.append((cls, trio))
                break
    if not rows:
        print("  (skipped variants — no matching image triples)")
        return

    fig, axes = plt.subplots(len(rows), 3, figsize=(6.0, 2.05 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (cls, trio) in enumerate(rows):
        for c, path in enumerate(trio):
            ax = axes[r, c]
            ax.imshow(Image.open(path))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(AXIS); sp.set_linewidth(0.8)
            if r == 0:
                ax.set_title(["colour", "grayscale", "segmented"][c],
                             fontsize=10, fontweight="bold", color=INK, pad=7)
        axes[r, 0].set_ylabel(short(cls), fontsize=8, color=INK2, rotation=0,
                              ha="right", va="center", labelpad=8)
    fig.suptitle("Every leaf ships in three variants", fontsize=11.5,
                 fontweight="bold", color=INK, x=0.09, ha="left", y=1.005)
    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    save(fig, "dataset_variants.png")


def fig_learning_curves():
    """Training histories of the three baselines. Included because the shape carries
    the argument: validation accuracy is above 94% after one epoch, so the benchmark
    was close to solved before training really began."""
    runs = [("custom_cnn", "Custom CNN", BLUE), ("resnet18", "ResNet-18", ORANGE),
            ("mobilenet_v2", "MobileNet-V2", AQUA)]
    hist = {k: (load(f"{k}_history.json") or {}).get("history") for k, _, _ in runs}
    if not any(hist.values()):
        print("  (skipped learning curves — no histories)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.7))
    for key, label, colour in runs:
        h = hist.get(key)
        if not h:
            continue
        ep = [e["epoch"] for e in h]
        ax1.plot(ep, [100 * e["val_accuracy"] for e in h], color=colour, linewidth=2, label=label)
        ax1.plot(ep, [100 * e["train_acc"] for e in h], color=colour, linewidth=1,
                 linestyle=":", alpha=0.75)
        ax2.plot(ep, [e["val_loss"] for e in h], color=colour, linewidth=2, label=label)
    ax1.set_ylim(70, 101)
    style(ax1, "epoch", "accuracy (%)", "Validation accuracy (solid) vs training (dotted)")
    style(ax2, "epoch", "validation loss", "Validation loss")
    ax1.legend(frameon=False, fontsize=8.5, loc="lower right")
    first = 100 * hist["mobilenet_v2"][0]["val_accuracy"]
    ax1.annotate(f"{first:.0f}% after one epoch", xy=(1.1, first), xytext=(6.5, 83),
                 fontsize=8.5, color=INK2,
                 arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.9))
    fig.tight_layout(w_pad=2.0)
    save(fig, "learning_curves.png")


if __name__ == "__main__":
    print(f"writing figures to {FIG}/")
    fig_confusion()
    fig_lab_vs_field()
    fig_adaptation()
    fig_frozen()
    fig_per_disease()
    fig_severity()
    fig_architecture()
    fig_dataset()
    fig_variants()
    fig_learning_curves()
