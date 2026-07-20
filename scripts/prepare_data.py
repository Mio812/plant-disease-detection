"""Download the datasets.

PlantVillage is the training corpus (colour, grayscale and segmented variants);
PlantDoc supplies the in-the-wild images used for the generalisation study.
Both are blob-filtered sparse clones, so only what is needed is fetched.

Usage:
    python -m scripts.prepare_data --dataset both
"""

import argparse
import subprocess
from pathlib import Path

PLANTVILLAGE = ("https://github.com/spMohanty/PlantVillage-Dataset.git", Path("data/PlantVillage"))
PLANTDOC = ("https://github.com/pratikkayal/PlantDoc-Dataset.git", Path("data/PlantDoc"))


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch PlantVillage and/or PlantDoc.")
    parser.add_argument("--dataset", choices=["plantvillage", "plantdoc", "both"], default="both")
    parser.add_argument("--variants", nargs="+", default=["raw/color", "raw/grayscale", "raw/segmented"],
                        help="PlantVillage subdirectories to check out (sparse-checkout is destructive: "
                             "listing fewer variants removes the others from the work tree)")
    parser.add_argument("--plantdoc-splits", nargs="+", default=["test", "train"])
    return parser.parse_args()


def sparse_clone(url, dest, subdirs):
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        url, str(dest)], check=True)
    present = [s for s in subdirs if (dest / s).exists()]
    if len(present) == len(subdirs):
        print(f"  {dest.as_posix()}: all requested variants already present, skipping sparse-checkout")
    else:
        subprocess.run(["git", "sparse-checkout", "set", *subdirs], cwd=str(dest), check=True)
    for sub in subdirs:
        n = sum(1 for _ in (dest / sub).rglob("*.*")) if (dest / sub).exists() else 0
        print(f"  {dest.as_posix()}/{sub}: {n} files")


def main():
    args = parse_args()
    if args.dataset in ("plantvillage", "both"):
        print("PlantVillage")
        sparse_clone(*PLANTVILLAGE, args.variants)
    if args.dataset in ("plantdoc", "both"):
        print("PlantDoc")
        sparse_clone(*PLANTDOC, args.plantdoc_splits)


if __name__ == "__main__":
    main()
