"""Show pipeline progress: which stages are done, what is running, and an ETA.

Reads configs/experiments.yaml for the declared stages and the newest run log
(PowerShell writes it as UTF-16, which is decoded here).

Usage:
    python -m scripts.status
    python -m scripts.status --watch
"""

import argparse
import glob
import re
import time
from datetime import datetime
from pathlib import Path

import yaml


def read_log(path):
    raw = Path(path).read_bytes()
    for encoding in ("utf-16", "utf-8", "cp1252"):
        try:
            text = raw.decode(encoding)
            if "[run]" in text or "COMP9444" in text:
                return text
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def latest_log():
    logs = sorted(glob.glob("logs/run_*.log"))
    return logs[-1] if logs else None


def stage_states(spec, text):
    ran = dict(re.findall(r"\[(ok|fail|blocked|done)\]\s+(\w+)", text or ""))
    started = re.findall(r"\[run\]\s+(\w+)", text or "")
    states = []
    for stage in spec["stages"] + spec.get("optional", []):
        sid = stage["id"]
        if Path(stage["produces"]).exists():
            state = "done"
        elif sid in ran:
            state = ran[sid]
        elif started and sid == started[-1]:
            state = "running"
        else:
            state = "pending"
        states.append((sid, stage.get("experiment", "-"), state))
    return states


def training_progress(text):
    epochs = re.findall(r"\[(\d+)/(\d+)\]\s+train_loss=([\d.]+)\s+train_acc=([\d.]+)"
                        r"\s+val_loss=([\d.]+)\s+val_acc=([\d.]+)", text or "")
    tags = re.findall(r"\[([\w.]+)\] variant=", text or "")
    rates = [float(r) for r in re.findall(r"(\d+\.\d+)it/s", text or "")]
    return (tags[-1] if tags else None), (epochs[-1] if epochs else None), (rates[-1] if rates else None)


def main():
    parser = argparse.ArgumentParser(description="Pipeline status.")
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    parser.add_argument("--watch", action="store_true", help="refresh every 20s")
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.experiments).read_text(encoding="utf-8"))

    while True:
        log = latest_log()
        text = read_log(log) if log else ""
        states = stage_states(spec, text)
        symbol = {"done": "[x]", "ok": "[x]", "running": "[>]",
                  "pending": "[ ]", "blocked": "[!]", "fail": "[F]"}

        print("\n" + "=" * 58)
        print(f"  {datetime.now():%H:%M:%S}   log: {Path(log).name if log else 'none'}")
        print("=" * 58)
        for sid, exp, state in states:
            print(f"  {symbol.get(state, '[?]')} {sid:22s} {exp:24s} {state}")

        tag, epoch, rate = training_progress(text)
        done = sum(1 for _, _, s in states if s in ("done", "ok"))
        if tag and epoch:
            cur, total = int(epoch[0]), int(epoch[1])
            print(f"\n  training : {tag}")
            print(f"  epoch    : {cur}/{total}   val_acc={epoch[5]}   train_acc={epoch[3]}")
            if rate:
                left = (total - cur) * 396 / rate / 60
                print(f"  speed    : {rate:.1f} it/s   ~{left:.0f} min left in this arm")
        remaining = [s for _, _, s in states if s in ("pending", "running")]
        print(f"\n  {done}/{len(states)} stages complete, {len(remaining)} to go")

        if not args.watch:
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
