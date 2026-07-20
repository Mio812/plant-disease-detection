"""Download the datasets.

PlantVillage is the training corpus (colour, grayscale and segmented variants);
PlantDoc supplies the in-the-wild images used for the generalisation study.
Both are blob-filtered sparse clones, so only what is needed is fetched.

Usage:
    python -m scripts.prepare_data --dataset both
"""

import argparse
import os
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


WINDOWS_ILLEGAL = ':?*"<>|'


def _windows_safe(dest, subdirs):
    """PlantDoc keeps URL query strings in its file names, which NTFS rejects.

    Git aborts the whole checkout on the first such path, so on Windows we list
    the offending blobs and exclude them with a sparse-checkout pattern rather
    than letting the stage fail.
    """
    listing = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD"],
                             cwd=str(dest), capture_output=True, text=True).stdout.splitlines()
    return [path for path in listing
            if path.split("/")[0] in subdirs
            and (any(c in WINDOWS_ILLEGAL for c in path.split("/")[-1])
                 or len(str(dest.resolve())) + len(path) > 250)]


def sparse_clone(url, dest, subdirs):
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        url, str(dest)], check=True)
    present = [s for s in subdirs if (dest / s).exists()]
    if len(present) == len(subdirs):
        print(f"  {dest.as_posix()}: all requested variants already present, skipping sparse-checkout")
    else:
        patterns = list(subdirs)
        unsafe = _windows_safe(dest, subdirs) if os.name == "nt" else []
        if unsafe:
            print(f"  skipping {len(unsafe)} files whose names are invalid on Windows")
            patterns += [f"!/{path}" for path in unsafe]
        result = subprocess.run(["git", "sparse-checkout", "set", "--no-cone", *patterns]
                                if unsafe else ["git", "sparse-checkout", "set", *subdirs],
                                cwd=str(dest))
        if result.returncode != 0:
            print(f"  WARNING: checkout of {dest.as_posix()} was incomplete; "
                  f"continuing with whatever was fetched")
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
