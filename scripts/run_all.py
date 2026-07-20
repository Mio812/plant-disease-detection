"""Run the experiment matrix declared in configs/experiments.yaml.

Stages are resumable: one whose `produces` file already exists is skipped, so an
interrupted run is continued by re-issuing the same command. Stage ids and their
`experiment` fields map onto docs/EXPERIMENTS.md.

Usage:
    python -m scripts.run_all --dry-run
    python -m scripts.run_all --quick            # 2 epochs, isolated outputs
    python -m scripts.run_all --optional         # include the extra ablations
    python -m scripts.run_all --only train_bg_random
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from src.config import Config


def expand(tokens, defaults):
    """Substitute ${...} placeholders; ${weights} expands into several arguments."""
    out = []
    for token in tokens:
        text = str(token)
        if text == "${weights}":
            out.extend(str(w) for w in defaults["weights"])
        elif text.startswith("${") and text.endswith("}"):
            out.append(str(defaults[text[2:-1]]))
        else:
            out.append(text)
    return out


def load_plan(path, include_optional):
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    stages = list(spec["stages"]) + (list(spec.get("optional", [])) if include_optional else [])
    return spec.get("defaults", {}), stages


def retarget(stages, out_dir, config_path):
    """Point paths and --config at the active output directory."""
    fixed = []
    for stage in stages:
        stage = dict(stage)
        if out_dir != "outputs":
            stage["produces"] = stage["produces"].replace("outputs/", f"{out_dir}/")
            stage["requires"] = [r.replace("outputs/", f"{out_dir}/")
                                 for r in stage.get("requires", [])]
            stage["run"] = [str(a).replace("outputs/", f"{out_dir}/") for a in stage["run"]]
        if stage["run"][0] != "prepare_data":
            stage["run"] = list(stage["run"]) + ["--config", config_path]
        fixed.append(stage)
    return fixed


def read(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def summarise(out_dir):
    rows = []
    lab = read(out_dir / "eval_plantvillage.json")
    seg = read(out_dir / "eval_segmented.json")
    field = read(out_dir / "eval_plantdoc.json")
    rows.append(["Baseline ensemble (128px)",
                 f"{lab['ensemble']:.2f}" if lab else "-",
                 f"{seg['ensemble']:.2f}" if seg else "-",
                 f"{field['ensemble']:.2f}" if field else "-"])

    for tag, label in [("resnet18_color_strong_p0_224", "Strong aug only (E8 control)"),
                       ("resnet18_color_strong_p70_224", "Background randomised p=0.7 (E8)"),
                       ("resnet18_color_strong_p100_224", "Background randomised p=1.0 (E8)"),
                       ("resnet18_segmented_strong_p0_224", "Trained on segmented (E9)"),
                       ("resnet18_grayscale_strong_p0_224", "Trained on grayscale (E10)"),
                       ("resnet50_color_strong_p70_224", "ResNet-50, p=0.7 (E8)")]:
        d = read(out_dir / f"{tag}_history.json")
        if not d:
            continue
        pv = d.get("test_metrics", {}).get("accuracy")
        col = f"{pv * 100:.2f}" if pv else "-"
        variant = d.get("variant", "color")
        rows.append([label,
                     "-" if variant != "color" else col,
                     col if variant == "segmented" else "-",
                     f"{d['plantdoc_accuracy']:.2f}" if d.get("plantdoc_accuracy") else "-"])

    for tag, label in [("ft_robust", "Fine-tuned 20-shot from robust (E11)"),
                       ("ft_baseline", "Fine-tuned 20-shot from baseline (E11)")]:
        d = read(out_dir / f"{tag}_history.json")
        if d:
            rows.append([label, "-", "-", f"{d['best']:.2f} (from {d['before']:.2f})"])

    header = ["Setting", "PlantVillage", "Segmented", "PlantDoc field"]
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(4)]
    lines = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += ["| " + " | ".join(r[i].ljust(widths[i]) for i in range(4)) + " |" for r in rows]

    extra = []
    probe = read(out_dir / "bias_probe.json")
    if probe:
        extra.append(f"E3 background-pixel probe: {probe['test_accuracy'] * 100:.1f}% "
                     f"(chance {probe['chance'] * 100:.1f}%)")
    cam = read(out_dir / "gradcam_audit.json")
    if cam:
        extra.append(f"E5 Grad-CAM inside leaf: {cam['mean_attention_in_leaf'] * 100:.1f}% "
                     f"vs {cam['mean_leaf_area_fraction'] * 100:.1f}% area "
                     f"({cam['attention_lift_over_area'] * 100:+.1f} pts)")
    full = read(out_dir / "eval_plantdoc_full.json")
    if full:
        extra.append(f"E6 zero-shot on all PlantDoc (n={full['n_images']}): "
                     f"ensemble {full['ensemble']:.2f}%")
    sev = read(out_dir / "severity_probe.json")
    if sev:
        extra.append(f"E13 severity AUC: {sev['auc_official_mask']:.3f} official mask, "
                     f"{sev['auc_otsu_mask']:.3f} Otsu")
    val = read(out_dir / "severity_validation.json")
    if val:
        extra.append(f"E14 severity vs manual: rho={val['spearman_rho']:.3f}, "
                     f"kappa={val['quadratic_kappa']:.3f}")

    text = "\n".join(lines) + ("\n\n" + "\n".join(extra) if extra else "") + "\n"
    (out_dir / "results_summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    parser = argparse.ArgumentParser(description="Run the declared experiment matrix.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--optional", action="store_true")
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--skip", nargs="+", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    defaults, stages = load_plan(args.experiments, args.optional and not args.quick)
    config_path = args.config
    out_dir = Path(Config.load(args.config).output_dir)
    if args.quick:
        defaults = {**defaults, "epochs": 2, "ft_epochs": 2}
        out_dir = Path("outputs_quick")
        out_dir.mkdir(parents=True, exist_ok=True)
        raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        raw["output_dir"] = out_dir.as_posix()
        config_path = (out_dir / "config_quick.yaml").as_posix()
        Path(config_path).write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        print(f"[quick]   isolated run -> {out_dir}/")
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = retarget(stages, out_dir.as_posix(), config_path)

    produced, results, started = set(), [], time.time()
    for stage in stages:
        name, produces = stage["id"], stage["produces"]
        produced.add(produces)
        if produces.endswith("_history.json"):
            produced.add(produces.replace("_history.json", "_best.pth"))
        if args.only and name not in args.only:
            continue
        if name in args.skip:
            print(f"[skip]    {name}")
            continue
        if Path(produces).exists() and not args.force:
            print(f"[done]    {name} -> {produces}")
            results.append((name, "cached", 0))
            continue
        missing = [r for r in stage.get("requires", [])
                   if not Path(r).exists() and not (args.dry_run and r in produced)]
        if missing:
            print(f"[blocked] {name}: missing {missing[0]}")
            results.append((name, "blocked", 0))
            continue
        argv = expand(stage["run"], defaults)
        cmd = [sys.executable, "-m", f"scripts.{argv[0]}", *argv[1:]]
        print(f"\n[run]     {name}  ({stage.get('experiment', '-')})\n          {' '.join(cmd[2:])}",
              flush=True)
        if args.dry_run:
            results.append((name, "dry-run", 0))
            continue
        t0 = time.time()
        code = subprocess.run(cmd).returncode
        dt = time.time() - t0
        results.append((name, "ok" if code == 0 else f"FAILED({code})", dt))
        print(f"[{'ok' if code == 0 else 'fail'}]      {name} in {dt / 60:.1f} min", flush=True)
        if code != 0 and args.stop_on_error:
            break

    print("\n" + "=" * 62 + "\nPIPELINE SUMMARY")
    for name, status, dt in results:
        print(f"  {name:22s} {status:12s} {dt / 60:6.1f} min")
    print(f"  total {(time.time() - started) / 60:.1f} min")
    if not args.dry_run:
        print("\n" + summarise(out_dir))
        if not (out_dir / "severity_validation.json").exists():
            print("Next: grade manual_grade (0-3) in outputs/severity_annotations.csv, then\n"
                  "  python -m scripts.audit --probe severity-validate")


if __name__ == "__main__":
    main()
