"""Run the whole experimental pipeline end to end, resumably.

Every stage declares the file it produces, so a stage whose output already
exists is skipped. The run can therefore be interrupted at any point and
restarted with the same command. When all stages have run, a comparison table
is written to ``outputs/results_summary.md``.

Usage:
    python -m scripts.run_all                     # full pipeline
    python -m scripts.run_all --quick             # 2-epoch smoke test
    python -m scripts.run_all --dry-run           # show the plan only
    python -m scripts.run_all --only robust_p70 ft_robust
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from src.config import Config

W = ["0.2", "0.5", "0.3"]          # validation-tuned ensemble weights


def stages(epochs, ft_epochs, full_ablation):
    plan = [
        ("download_plantdoc", "data/PlantDoc/train", [],
         ["download_plantdoc", "--split", "both"]),
        ("bias_probe", "outputs/bias_probe.json", [],
         ["bias_probe"]),
        ("eval_segmented", "outputs/segmented_results.json", ["outputs/resnet18_best.pth"],
         ["eval_segmented", "--weights", *W]),
        ("plantdoc_full", "outputs/plantdoc_full.json", ["data/PlantDoc/train"],
         ["eval_plantdoc", "--weights", *W, "--plantdoc", "data/PlantDoc/test",
          "data/PlantDoc/train", "--out", "outputs/plantdoc_full.json"]),
        ("plantdoc_test", "outputs/plantdoc_test.json", ["data/PlantDoc/test"],
         ["eval_plantdoc", "--weights", *W, "--plantdoc", "data/PlantDoc/test",
          "--out", "outputs/plantdoc_test.json"]),
        ("robust_p0", "outputs/resnet18_color_p0_224_history.json", [],
         ["train_robust", "--model", "resnet18", "--image-size", "224",
          "--p-random", "0.0", "--epochs", str(epochs)]),
        ("robust_p70", "outputs/resnet18_color_p70_224_history.json", [],
         ["train_robust", "--model", "resnet18", "--image-size", "224",
          "--p-random", "0.7", "--epochs", str(epochs)]),
        ("robust_segmented", "outputs/resnet18_segmented_p0_224_history.json", [],
         ["train_robust", "--model", "resnet18", "--image-size", "224",
          "--variant", "segmented", "--epochs", str(epochs)]),
    ]
    if full_ablation:
        plan += [
            ("robust_p100", "outputs/resnet18_color_p100_224_history.json", [],
             ["train_robust", "--model", "resnet18", "--image-size", "224",
              "--p-random", "1.0", "--epochs", str(epochs)]),
            ("robust_r50", "outputs/resnet50_color_p70_224_history.json", [],
             ["train_robust", "--model", "resnet50", "--image-size", "224",
              "--p-random", "0.7", "--epochs", str(epochs)]),
        ]
    plan += [
        ("ft_robust", "outputs/ft_robust_history.json",
         ["outputs/resnet18_color_p70_224_best.pth", "data/PlantDoc/train"],
         ["finetune_plantdoc", "--model", "resnet18", "--image-size", "224",
          "--checkpoint", "outputs/resnet18_color_p70_224_best.pth",
          "--shots", "20", "--epochs", str(ft_epochs), "--tag", "ft_robust"]),
        ("ft_baseline", "outputs/ft_baseline_history.json",
         ["outputs/resnet18_best.pth", "data/PlantDoc/train"],
         ["finetune_plantdoc", "--model", "resnet18",
          "--checkpoint", "outputs/resnet18_best.pth",
          "--shots", "20", "--epochs", str(ft_epochs), "--tag", "ft_baseline"]),
        ("severity_sample", "outputs/severity_annotations.csv", [],
         ["severity_sample", "--n", "150"]),
    ]
    return plan


def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def summarise(out_dir):
    rows = []

    def acc(d, *keys):
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return None
        return d if isinstance(d, (int, float)) else None

    ens = load(out_dir / "ensemble_history.json")
    seg = load(out_dir / "segmented_results.json")
    pdt = load(out_dir / "plantdoc_test.json")
    pdf = load(out_dir / "plantdoc_full.json")
    base_pv = acc(ens, "test_metrics", "accuracy")
    rows.append(["Baseline ensemble (128px)",
                 f"{base_pv * 100:.2f}" if base_pv else "-",
                 f"{seg['ensemble']:.2f}" if seg else "-",
                 f"{pdt['ensemble']:.2f}" if pdt else "-"])

    for tag, label in [("resnet18_color_p0_224", "Strong aug only (p=0.0)"),
                       ("resnet18_color_p70_224", "Background randomised (p=0.7)"),
                       ("resnet18_segmented_p0_224", "Trained on segmented"),
                       ("resnet18_color_p100_224", "Background randomised (p=1.0)"),
                       ("resnet50_color_p70_224", "ResNet-50, p=0.7")]:
        d = load(out_dir / f"{tag}_history.json")
        if not d:
            continue
        pv = acc(d, "test_metrics", "accuracy")
        col = f"{pv * 100:.2f}" if pv else "-"
        seg_col = col if "segmented" in tag else "-"
        pv_col = "-" if "segmented" in tag else col
        rows.append([label, pv_col, seg_col, f"{d.get('plantdoc_accuracy', float('nan')):.2f}"])

    for tag, label in [("ft_robust", "Fine-tuned 20-shot (from robust)"),
                       ("ft_baseline", "Fine-tuned 20-shot (from baseline)")]:
        d = load(out_dir / f"{tag}_history.json")
        if d:
            rows.append([label, "-", "-", f"{d['best']:.2f}  (from {d['before']:.2f})"])

    header = ["Setting", "PlantVillage colour", "PlantVillage segmented", "PlantDoc field"]
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(4)]
    lines = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += ["| " + " | ".join(r[i].ljust(widths[i]) for i in range(4)) + " |" for r in rows]

    probe = load(out_dir / "bias_probe.json")
    extra = []
    if probe:
        extra.append(f"Background-pixel probe: {probe['test_accuracy'] * 100:.1f}% "
                     f"(chance {100 / 38:.1f}%)")
    if pdf:
        extra.append(f"Zero-shot on all PlantDoc (n={pdf.get('n_images', '?')}): "
                     f"ensemble {pdf['ensemble']:.2f}%")
    sev = load(out_dir / "severity_validation.json")
    if sev:
        extra.append(f"Severity vs manual grades: rho={sev['spearman_rho']:.3f}, "
                     f"kappa={sev['quadratic_kappa']:.3f}")

    text = "\n".join(lines) + ("\n\n" + "\n".join(extra) if extra else "") + "\n"
    (out_dir / "results_summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    parser = argparse.ArgumentParser(description="Run the full experiment pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ft-epochs", type=int, default=15)
    parser.add_argument("--quick", action="store_true", help="2-epoch smoke test, core stages only")
    parser.add_argument("--full-ablation", action="store_true", help="also run p=1.0 and ResNet-50")
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--skip", nargs="+", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.epochs, args.ft_epochs = 2, 2
    out_dir = Path(Config.load(args.config).output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = stages(args.epochs, args.ft_epochs, args.full_ablation and not args.quick)

    produced_earlier = set()
    results, started = [], time.time()
    for name, produces, requires, argv in plan:
        produced_earlier.add(produces)
        if produces.endswith("_history.json"):
            produced_earlier.add(produces.replace("_history.json", "_best.pth"))
        if args.only and name not in args.only:
            continue
        if name in args.skip:
            print(f"[skip]    {name} (requested)")
            continue
        if Path(produces).exists() and not args.force:
            print(f"[done]    {name} -> {produces} already exists")
            results.append((name, "cached", 0))
            continue
        missing = [r for r in requires
                   if not Path(r).exists() and not (args.dry_run and r in produced_earlier)]
        if missing:
            print(f"[blocked] {name}: missing {missing[0]}")
            results.append((name, "blocked", 0))
            continue
        cmd = [sys.executable, "-m", f"scripts.{argv[0]}", *argv[1:]]
        print(f"\n[run]     {name}\n          {' '.join(cmd[2:])}", flush=True)
        t0 = time.time()
        if args.dry_run:
            results.append((name, "dry-run", 0))
            continue
        code = subprocess.run(cmd).returncode
        dt = time.time() - t0
        results.append((name, "ok" if code == 0 else f"FAILED({code})", dt))
        print(f"[{'ok' if code == 0 else 'fail'}]      {name} in {dt / 60:.1f} min", flush=True)
        if code != 0 and args.stop_on_error:
            break

    print("\n" + "=" * 60 + "\nPIPELINE SUMMARY")
    for name, status, dt in results:
        print(f"  {name:20s} {status:12s} {dt / 60:6.1f} min")
    print(f"  total {(time.time() - started) / 60:.1f} min")
    if not args.dry_run:
        print("\n" + summarise(out_dir))
        print(f"written to {(out_dir / 'results_summary.md').as_posix()}")
        if not (out_dir / "severity_validation.json").exists():
            print("\nNext: grade `manual_grade` (0-3) in outputs/severity_annotations.csv, then run\n"
                  "  python -m scripts.severity_validate --csv outputs/severity_annotations.csv")


if __name__ == "__main__":
    main()
