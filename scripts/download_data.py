"""Fetch the PlantVillage colour images into ``data/``.

Uses a partial + sparse clone of the public PlantVillage repository so only the
colour image set (``raw/color``) is downloaded, not the grayscale/segmented
variants. The result is exposed at ``data/PlantVillage/raw/color``, matching the
default ``data.root`` in ``configs/default.yaml``.

Usage:
    python -m scripts.download_data
"""

import subprocess
from pathlib import Path

REPO_URL = "https://github.com/spMohanty/PlantVillage-Dataset.git"
DEST = Path("data/PlantVillage")
SUBDIR = "raw/color"


def _run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


def main():
    color_dir = DEST / SUBDIR
    if color_dir.exists():
        print(f"{color_dir.as_posix()} already exists; skipping clone.")
        return

    DEST.parent.mkdir(parents=True, exist_ok=True)
    _run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO_URL, str(DEST))
    _run("git", "sparse-checkout", "set", SUBDIR, cwd=str(DEST))

    if color_dir.exists():
        n_classes = sum(1 for p in color_dir.iterdir() if p.is_dir())
        print(f"Done: {n_classes} classes at {color_dir.as_posix()} (matches data.root).")
    else:
        print("Clone finished but raw/color was not found; check the repository layout.")


if __name__ == "__main__":
    main()
