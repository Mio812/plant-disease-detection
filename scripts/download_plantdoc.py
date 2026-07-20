"""Fetch the PlantDoc test split (field images) into ``data/PlantDoc``.

A blob-filtered sparse clone keeps the download to the ~236-image test split
instead of the full ~1 GB repository.

Usage:
    python -m scripts.download_plantdoc
"""

import subprocess
from pathlib import Path

REPO = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
DEST = Path("data/PlantDoc")


def main():
    if (DEST / "test").exists():
        print(f"{DEST.as_posix()}/test already exists; skipping.")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                    REPO, str(DEST)], check=True)
    subprocess.run(["git", "sparse-checkout", "set", "test"], cwd=str(DEST), check=True)
    n = sum(1 for _ in (DEST / "test").rglob("*.*"))
    print(f"Done: {n} field images at {(DEST / 'test').as_posix()}")


if __name__ == "__main__":
    main()
