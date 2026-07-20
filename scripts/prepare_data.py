"""Download PlantVillage (training corpus) and PlantDoc (field images).

Both are blob-filtered sparse clones. Some PlantDoc file names keep URL query
strings and contain characters NTFS forbids; those blobs are excluded on Windows,
and completeness is checked against the repository tree so an interrupted
checkout resumes.

Usage:
    python -m scripts.prepare_data --dataset both
"""

import argparse
import os
import subprocess
from pathlib import Path

PLANTVILLAGE = ("https://github.com/spMohanty/PlantVillage-Dataset.git", Path("data/PlantVillage"))
PLANTDOC = ("https://github.com/pratikkayal/PlantDoc-Dataset.git", Path("data/PlantDoc"))
WINDOWS_ILLEGAL = ':?*"<>|'
MAX_PATH = 250


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch PlantVillage and/or PlantDoc.")
    parser.add_argument("--dataset", choices=["plantvillage", "plantdoc", "both"], default="both")
    parser.add_argument("--variants", nargs="+",
                        default=["raw/color", "raw/grayscale", "raw/segmented"],
                        help="PlantVillage subdirectories to check out")
    parser.add_argument("--plantdoc-splits", nargs="+", default=["test", "train"])
    parser.add_argument("--force", action="store_true", help="re-run checkout even if complete")
    return parser.parse_args()


def tree(dest):
    return subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD"],
                          cwd=str(dest), capture_output=True, text=True).stdout.splitlines()


def checkoutable(dest, paths):
    """Paths git can actually write on this platform."""
    if os.name != "nt":
        return paths, []
    root = len(str(dest.resolve())) + 1
    unsafe = [p for p in paths
              if any(c in WINDOWS_ILLEGAL for c in p.split("/")[-1]) or root + len(p) > MAX_PATH]
    return [p for p in paths if p not in set(unsafe)], unsafe


def sparse_clone(url, dest, subdirs, force=False):
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        url, str(dest)], check=True)

    listing = [p for p in tree(dest) if p.split("/")[0] in subdirs]
    wanted, unsafe = checkoutable(dest, listing)
    on_disk = sum(1 for sub in subdirs for _ in (dest / sub).rglob("*.*")) if any(
        (dest / sub).exists() for sub in subdirs) else 0

    sentinel = dest / ".complete"
    if on_disk >= len(wanted) and not force:
        sentinel.write_text(str(on_disk), encoding="utf-8")
        print(f"  {dest.as_posix()}: complete ({on_disk} files), skipping checkout")
    else:
        if on_disk:
            print(f"  {dest.as_posix()}: incomplete ({on_disk} of {len(wanted)}), resuming")
        if unsafe:
            print(f"  excluding {len(unsafe)} files whose names are invalid on this platform")
            patterns = [f"/{sub}/" for sub in subdirs] + [f"!/{p}" for p in unsafe]
            result = subprocess.run(["git", "sparse-checkout", "set", "--no-cone", *patterns],
                                    cwd=str(dest))
        else:
            result = subprocess.run(["git", "sparse-checkout", "set", *subdirs], cwd=str(dest))
        if result.returncode != 0:
            print(f"  WARNING: checkout returned {result.returncode}; continuing with what arrived")
        final = sum(1 for sub in subdirs for _ in (dest / sub).rglob("*.*"))
        if final >= len(wanted):
            sentinel.write_text(str(final), encoding="utf-8")
            print(f"  {dest.as_posix()}: now complete ({final} files)")
        else:
            sentinel.unlink(missing_ok=True)
            print(f"  {dest.as_posix()}: STILL INCOMPLETE ({final} of {len(wanted)}) - re-run to resume")

    for sub in subdirs:
        n = sum(1 for _ in (dest / sub).rglob("*.*")) if (dest / sub).exists() else 0
        classes = len([d for d in (dest / sub).iterdir() if d.is_dir()]) if (dest / sub).exists() else 0
        print(f"  {dest.as_posix()}/{sub}: {n} files"
              + (f", {classes} classes" if classes else ""))


def main():
    args = parse_args()
    if args.dataset in ("plantvillage", "both"):
        print("PlantVillage")
        sparse_clone(*PLANTVILLAGE, args.variants, args.force)
    if args.dataset in ("plantdoc", "both"):
        print("PlantDoc")
        sparse_clone(*PLANTDOC, args.plantdoc_splits, args.force)


if __name__ == "__main__":
    main()
