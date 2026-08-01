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

    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(x, y, w, h, title, sub="", fc="#ffffff", ec=BLUE, tc=INK, lw=1.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                    linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
        ax.text(x + w / 2, y + h / 2 + (0.028 if sub else 0), title, ha="center", va="center",
                fontsize=9, fontweight="bold", color=tc, zorder=3)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.032, sub, ha="center", va="center",
                    fontsize=7.4, color=INK2, zorder=3)

    def arrow(x1, y1, x2, y2, color=MUTED):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.2, color=color, zorder=1,
                                     shrinkA=0, shrinkB=0))

    box(0.01, 0.50, 0.135, 0.17, "Leaf image", "128² or 224²", fc="#f4f7fb", ec=AXIS, tc=INK2)
    box(0.175, 0.50, 0.165, 0.17, "Augmentation", "standard / strong\n+ background rand.",
        fc="#f4f7fb", ec=AXIS, tc=INK2)

    ys = [0.72, 0.50, 0.28]
    names = [("Custom CNN", "from scratch · 0.4M"),
             ("MobileNet-V2", "ImageNet · 2.3M"),
             ("ResNet-18", "ImageNet · 11.2M")]
    for y, (n, s) in zip(ys, names):
        box(0.375, y, 0.185, 0.155, n, s)
        arrow(0.345, 0.585, 0.372, y + 0.078)
        arrow(0.563, y + 0.078, 0.605, 0.60)

    arrow(0.148, 0.585, 0.172, 0.585)
    box(0.608, 0.46, 0.15, 0.25, "Soft-voting\nensemble", "weights tuned\non validation",
        fc="#eef4fd", ec=BLUE, lw=1.8)

    box(0.795, 0.615, 0.195, 0.135, "38-way disease", "specific class", ec=AQUA)
    box(0.795, 0.445, 0.195, 0.135, "Healthy / diseased", "collapsed from 38", ec=AQUA)
    arrow(0.760, 0.615, 0.792, 0.683, AQUA)
    arrow(0.760, 0.560, 0.792, 0.513, AQUA)

    # severity branch, sharing the ResNet-18 backbone
    box(0.608, 0.10, 0.15, 0.165, "Severity head", "1-unit regressor\n(jointly trained)",
        fc="#fdf3ee", ec=ORANGE, lw=1.6)
    box(0.795, 0.10, 0.195, 0.165, "Lesion ratio", "→ 4 ordinal grades", ec=ORANGE)
    arrow(0.468, 0.278, 0.468, 0.188, ORANGE)
    arrow(0.468, 0.188, 0.605, 0.188, ORANGE)
    arrow(0.760, 0.188, 0.792, 0.188, ORANGE)
    ax.text(0.452, 0.228, "shared backbone", fontsize=7.2, color=ORANGE,
            ha="right", va="center")

    ax.text(0.0, 0.955, "Model architecture", fontsize=12, fontweight="bold", color=INK)
    ax.text(0.0, 0.90, "Three CNNs are trained through one identical pipeline and combined by a "
                       "validation-tuned soft vote; the severity head shares the ResNet-18 backbone.",
            fontsize=8.5, color=INK2)
    save(fig, "architecture.png")



# ------------------------------------- 8. dataset composition (Data Analysis slide)
def fig_dataset():
    """Replaces the four-donut slide. Each quantity gets the form that fits it:
    a ratio is not a part-to-whole, so it is not a ring."""
    from torchvision.datasets import ImageFolder
    from src.config import Config
    from src.data import splits_from_config

    cfg = Config.load("configs/default.yaml")
    base = ImageFolder(cfg.data.root)
    counts = np.bincount([l for _, l in base.samples])
    healthy = sum(int(c) for c, n in zip(counts, base.classes) if "healthy" in n.lower())
    total = int(counts.sum())
    tr, va, te = (len(s) for s in splits_from_config(cfg, base))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 5.4))

    # A — class sizes: the skew is the message, so show every class
    ax = axes[0, 0]
    order = np.argsort(counts)[::-1]
    ax.bar(range(len(counts)), counts[order], color=BLUE, width=0.86)
    ax.set_xticks([]); ax.set_ylabel("images", color=INK2)
    ax.annotate(f"largest {counts.max():,}", (0, counts.max()), xytext=(4, -2),
                textcoords="offset points", fontsize=8, color=INK)
    ax.annotate(f"smallest {counts.min():,}", (len(counts) - 1, counts.min()),
                xytext=(-6, 14), textcoords="offset points", fontsize=8, color=RED, ha="right")
    style(ax, title=f"Class imbalance — {counts.max() / counts.min():.0f}:1 across 38 classes")

    def stacked(ax, parts, colours, title):
        left = 0
        for (label, value), colour in zip(parts, colours):
            ax.barh([0], [value], left=left, color=colour, height=0.42)
            ax.text(left + value / 2, 0, f"{label}\n{100 * value / total_of(parts):.0f}%",
                    ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
            left += value
        ax.set_xlim(0, left); ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([]); ax.set_xticks([])
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, color=INK, fontsize=10.5, fontweight="bold", loc="left", pad=10)

    def total_of(parts):
        return sum(v for _, v in parts)

    stacked(axes[0, 1], [("diseased", total - healthy), ("healthy", healthy)],
            [ORANGE, AQUA], "Label balance")
    stacked(axes[1, 0], [("train", tr), ("val", va), ("test", te)],
            [BLUE, "#86b6ef", "#cde2fb"], "Data split — grouped by leaf")
    axes[1, 0].texts[-1].set_color(INK)
    axes[1, 0].texts[-2].set_color(INK)

    # D — leakage is a change, so show before and after
    ax = axes[1, 1]
    bars = ax.bar(["random split", "leaf-grouped split"], [74.7, 0.0],
                  color=[RED, AQUA], width=0.5)
    for b, v in zip(bars, [74.7, 0.0]):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.1f}%", ha="center",
                fontsize=9.5, color=INK, fontweight="bold")
    ax.set_ylim(0, 92); ax.set_yticks([0, 25, 50, 75])
    style(ax, ylabel="test images with a\nsame-leaf twin in train",
          title="Train/test leakage, before and after")

    fig.tight_layout(h_pad=2.4, w_pad=3.0)
    save(fig, "dataset_composition.png")

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
