"""Cross-check every claim in the deliverables against the committed artefacts."""
import json, re, sys
from pathlib import Path

def g(f):
    p = Path("outputs") / f
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

lab, field = g("eval_plantvillage.json"), g("eval_plantdoc.json")
full, seg = g("eval_plantdoc_full.json"), g("eval_segmented.json")
val, joint = g("severity_validation.json"), g("joint_severity.json")
sev, bias, cam = g("severity_probe.json"), g("bias_probe.json"), g("gradcam_audit.json")
res = g("residual_leakage.json")
pairs = g("ensemble_plantvillage_confused_pairs.json")
h = lambda s: "healthy" in s.lower()
cross = sum(p["count"] for p in pairs.get("pairs", []) if h(p["true"]) != h(p["pred"]))

def hist(tag, key="test_metrics"):
    d = g(f"{tag}_history.json")
    return round(d[key]["accuracy"] * 100, 2) if d.get(key) else None

TRUTH = {
    "ensemble 38-way": lab.get("ensemble"),
    "ensemble binary": lab.get("ensemble_binary"),
    "test n": lab.get("n_images"),
    "errors": pairs.get("total_errors"),
    "crossings": cross,
    "custom_cnn": hist("custom_cnn"), "resnet18": hist("resnet18"),
    "mobilenet_v2": hist("mobilenet_v2"),
    "segmented-trained": hist("resnet18_segmented_strong_p0_224"),
    "grayscale-trained": hist("resnet18_grayscale_strong_p0_224"),
    "segmented zero-shot": seg.get("ensemble"),
    "field 236": field.get("ensemble"), "field binary": field.get("ensemble_binary"),
    "field 2525": full.get("ensemble"),
    "E3 bg probe": round(bias.get("test_accuracy", 0) * 100, 1),
    "E5 gradcam": round(cam.get("mean_attention_in_leaf", 0) * 100, 1),
    "E5 leaf area": round(cam.get("mean_leaf_area_fraction", 0) * 100, 1),
    "residual leak %": res.get("test_flagged_pct"),
    "E13 official AUC": sev.get("auc_official_mask"), "E13 otsu AUC": sev.get("auc_otsu_mask"),
    "E14 annotators": val.get("annotators"),
    "E14 ceiling": round(val.get("inter_annotator_kappa_mean_pairwise", 0), 3),
    "E14 kappa raw": round(val.get("quadratic_kappa", 0), 3),
    "E14 kappa cal": round(val.get("quadratic_kappa_recalibrated_cv", 0), 3),
    "E26 rho head": joint.get("within_class_rho_head"),
    "E26 rho otsu": joint.get("within_class_rho_otsu"),
    "E26 class acc": joint.get("classification_accuracy"),
    "ft 5": g("ft_shots5_history.json").get("best"),
    "ft 20": g("ft_robust_history.json").get("best"),
    "ft 100": g("ft_shots100_history.json").get("best"),
    "ft all": g("ft_full_history.json").get("best"),
    "frozen lab": hist("resnet18_color_strong_p0_224_frozen"),
    "full lab": hist("resnet18_color_strong_p0_224"),
    "frozen field": g("eval_arm_frozen_ctrl.json").get("ensemble"),
    "full field": g("eval_arm_bg_control.json").get("ensemble"),
    "bg_random crop": g("eval_arm_bg_random.json").get("ensemble_crop"),
}
print("=== GROUND TRUTH (from outputs/) ===")
for k, v in TRUTH.items():
    print(f"  {k:22s} {v}")

# ---- patterns that must NOT appear anywhere (superseded / contradicted) ----
STALE = {
    r"99\.84": "leaky ensemble 38-way (now 99.53)",
    r"\b17\.37\b": "leaky field 236 (now 16.10)",
    r"\b24\.15\b": "leaky frozen field (now 16.91)",
    r"\b48\.73\b": "leaky 20-shot (now 47.03)",
    r"\b56\.78\b": "leaky segmented (now 70.05)",
    r"\b8,145\b": "leaky test size (now 8,215)",
    r"\b91\.70\b": "leaky frozen lab (now 91.26)",
    r"no error crosses|none crossing|never confuses healthy|no image crosses": "false: 3 crossings",
    r"honest 91%": "wrong source (now 93%)",
    r"13 / 8,145|13 residual": "leaky error count (now 39)",
    r"\b100\.00%\b": "binary is 99.96, not a perfect 100",
}
DELIVERABLES = ["README.md", "docs/EXPERIMENTS.md", "docs/PLAN.md", "docs/DEVELOPMENT.md",
                "report/update_ppt.py", "tools/build_notebook.py",
                "outputs/results_summary.md"]


def office_text(path):
    """Paragraph text out of a .docx / .pptx, so hand-edited files are audited too."""
    import zipfile, re as _re
    parts = {".docx": ["word/document.xml"], ".pptx": None}
    z = zipfile.ZipFile(path)
    names = (parts[path.suffix] if parts.get(path.suffix)
             else [n for n in z.namelist() if n.startswith("ppt/slides/slide")])
    out = []
    for n in names:
        xml = z.read(n).decode("utf-8", "replace")
        for para in _re.findall(r"<a:p>.*?</a:p>|<w:p[ >].*?</w:p>", xml, _re.S):
            txt = "".join(_re.findall(r"<(?:a|w):t[^>]*>(.*?)</(?:a|w):t>", para, _re.S))
            if txt.strip():
                out.append(txt)
    return out
print("\n=== STALE / CONTRADICTORY PATTERNS ===")
issues = 0
for f in DELIVERABLES:
    p = Path(f)
    if not p.exists():
        continue
    text = p.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(text.splitlines(), 1):
        for pat, why in STALE.items():
            if re.search(pat, line, re.I):
                # allow explicit leaky-vs-honest comparisons
                if re.search(r"leak|leaked|was |random split|before|inflat|superseded", line, re.I):
                    continue
                print(f"  {f}:{i}  [{why}]")
                print(f"      {line.strip()[:110]}")
                issues += 1
print(f"\n  {issues} suspicious line(s)")
