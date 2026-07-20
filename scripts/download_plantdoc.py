"""Fetch PlantDoc field images into ``data/PlantDoc``.

A blob-filtered sparse clone avoids pulling the whole ~1 GB repository. The
test split (~236 images) is enough for zero-shot evaluation; add ``--split
train`` for the images used by ``scripts.finetune_plantdoc``.

Usage:
    python -m scripts.download_plantdoc --split both
"""

import argparse
import subprocess
from pathlib import Path

REPO = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
DEST = Path("data/PlantDoc")


def parse_args():
    parser = argparse.ArgumentParser(description="Download PlantDoc splits.")
    parser.add_argument("--split", choices=["test", "train", "both"], default="test")
    return parser.parse_args()


def main():
    args = parse_args()
    splits = ["test", "train"] if args.split == "both" else [args.split]

    if not (DEST / ".git").exists():
        DEST.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        REPO, str(DEST)], check=True)
    subprocess.run(["git", "sparse-checkout", "set", *splits], cwd=str(DEST), check=True)

    for split in splits:
        n = sum(1 for _ in (DEST / split).rglob("*.*"))
        print(f"{split}: {n} images at {(DEST / split).as_posix()}")


if __name__ == "__main__":
    main()
