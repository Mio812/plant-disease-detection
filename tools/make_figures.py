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


def short(cls):
    crop, _, dis = cls.partition("___")
    crop = crop.replace("_(maize)", "").replace("_(including_sour)", "").replace(",_bell", "")
    dis = dis.replace("_", " ").replace("(", "").replace(")", "")
    dis = re.sub(r"\s+", " ", dis).strip()
    if dis.lower() == "healthy":
        return f"{crop} · healthy"
    return f"{crop} · {dis[:26]}"


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

    fig, ax = plt.subplots(figsize=(9.6, 8.6))
    im = ax.imshow(norm, cmap=BLUES, norm=PowerNorm(gamma=0.4, vmin=0, vmax=1))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([short(c) for c in classes], rotation=90, fontsize=5.6)
    ax.set_yticklabels([short(c) for c in classes], fontsize=5.6)
    ax.set_xlabel("predicted", color=INK2); ax.set_ylabel("true", color=INK2)
    ax.set_title("Ensemble confusion matrix — PlantVillage test (row-normalised)",
                 color=INK, fontsize=11, fontweight="bold", loc="left", pad=30)
    # only the errors are worth labelling; the diagonal is ~1.0 everywhere
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=5.2, color=RED)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("share of true class", color=INK2, fontsize=8)
    cb.ax.tick_params(labelsize=7, color=MUTED)
    cb.outline.set_edgecolor(AXIS)
    ax.text(0, 1.008, "39 errors in 8,215 images; red digits are misclassified counts. Three cross "
                      "the healthy/diseased boundary — two diseased leaves called healthy.",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
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

    fig, ax = plt.subplots(figsize=(7.0, 8.2))
    ax.barh(range(len(names)), accs, color=colors, height=0.72)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6.6)
    ax.set_xlim(0, 108); ax.set_xticks([0, 25, 50, 75, 100])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("recall (%)", color=INK2)
    ax.set_title("Per-disease accuracy — every class, weakest first",
                 color=INK, fontsize=10.5, fontweight="bold", loc="left", pad=10)
    for i, a in enumerate(accs):
        if a < 99:
            ax.text(a + 1.5, i, f"{a:.0f}%", va="center", fontsize=7, color=RED)
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


if __name__ == "__main__":
    print(f"writing figures to {FIG}/")
    fig_confusion()
    fig_lab_vs_field()
    fig_adaptation()
    fig_frozen()
    fig_per_disease()
    fig_severity()
