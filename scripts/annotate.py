"""Package the severity annotation task for a team, and merge the results back.

`--action package` copies the sampled leaves into one self-contained folder per
annotator, each with its own CSV, so teammates need neither the repository nor
the dataset. `--action merge` reads the returned CSVs back into the master file.

Usage:
    python -m scripts.annotate --action package --annotators 4
    python -m scripts.annotate --action merge
"""

import argparse
import csv
import shutil
from pathlib import Path

from src.config import Config

INSTRUCTIONS = """Severity grading - COMP9444 Project 090
=======================================

Open each image in the `images` folder and grade how much of the LEAF AREA
shows disease symptoms (lesions, spots, discolouration, necrosis).

Write one number in the `manual_grade` column of grades.csv:

    0 = healthy     no visible symptoms
    1 = mild        symptoms on roughly < 5% of the leaf
    2 = moderate    roughly 5-20% of the leaf
    3 = severe      more than about 20% of the leaf

Guidance
- Judge the proportion of the LEAF, not the whole image.
- Ignore the background entirely.
- These are all leaves the classifier called diseased, so 0 is possible but
  should be uncommon - use it when you genuinely see no symptoms.
- If you cannot tell, leave the cell empty rather than guessing.
- Do not change any other column, and do not rename files.

Send grades.csv back when done.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Package or merge severity annotations.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--action", choices=["package", "merge"], required=True)
    parser.add_argument("--annotators", type=int, default=4)
    parser.add_argument("--master", default="outputs/severity_annotations.csv")
    parser.add_argument("--dir", default="outputs/annotation")
    return parser.parse_args()


def package(args):
    rows = list(csv.DictReader(open(args.master, encoding="utf-8")))
    root = Path(args.dir)
    if root.exists():
        shutil.rmtree(root)
    per = -(-len(rows) // args.annotators)
    for k in range(args.annotators):
        shard = rows[k * per:(k + 1) * per]
        if not shard:
            continue
        folder = root / f"annotator_{k + 1}"
        (folder / "images").mkdir(parents=True, exist_ok=True)
        (folder / "INSTRUCTIONS.txt").write_text(INSTRUCTIONS, encoding="utf-8")
        with (folder / "grades.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "class", "manual_grade"])
            for row in shard:
                source = Path(row["path"])
                name = f"{Path(row['path']).stem}.jpg"
                shutil.copy2(source, folder / "images" / name)
                writer.writerow([name, row["class"], ""])
        print(f"  {folder.as_posix()}: {len(shard)} images")
    print(f"\nZip each annotator_N folder and send it. They fill `manual_grade` (0-3)\n"
          f"in grades.csv and return it. Then: python -m scripts.annotate --action merge")


def merge(args):
    root = Path(args.dir)
    grades = {}
    for csv_path in sorted(root.glob("annotator_*/grades.csv")):
        for row in csv.DictReader(open(csv_path, encoding="utf-8")):
            value = row.get("manual_grade", "").strip()
            if value:
                grades[row["image"]] = value
        print(f"  read {csv_path.as_posix()}")

    rows = list(csv.DictReader(open(args.master, encoding="utf-8")))
    filled = 0
    for row in rows:
        key = f"{Path(row['path']).stem}.jpg"
        if key in grades:
            row["manual_grade"] = grades[key]
            filled += 1
    with open(args.master, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nmerged {filled}/{len(rows)} grades into {args.master}")
    print("Next: python -m scripts.audit --probe severity-validate")


def main():
    args = parse_args()
    Config.load(args.config)
    (package if args.action == "package" else merge)(args)


if __name__ == "__main__":
    main()
