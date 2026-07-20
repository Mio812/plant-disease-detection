"""Package the severity annotation task for a team and merge the results.

``package`` writes one self-contained folder per annotator (images, a CSV and a
browser grader), so teammates need neither the repository nor the dataset.
``merge`` reads the returned CSVs back into the master file, matching on image
name rather than row order.

Usage:
    python -m scripts.annotate --action package --annotators 4
    python -m scripts.annotate --action merge
"""

import argparse
import csv
import json
import shutil
from pathlib import Path, PurePath

from src.config import Config


GRADER_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Severity grading</title>
<style>
 body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#14200f;color:#eef3e8;
      display:flex;flex-direction:column;align-items:center;min-height:100vh}
 header{width:100%;background:#1e3318;padding:10px 18px;box-sizing:border-box}
 h1{font-size:16px;margin:0 0 6px}
 #bar{height:6px;background:#33502a;border-radius:3px;overflow:hidden}
 #fill{height:100%;width:0;background:#8FB55E;transition:width .2s}
 #meta{font-size:13px;color:#b9cfa8;margin-top:6px}
 img{max-width:min(70vw,560px);max-height:56vh;border-radius:8px;margin:14px 0;background:#000}
 .btns{display:flex;gap:10px;flex-wrap:wrap;justify-content:center}
 button{font-size:15px;padding:10px 16px;border:0;border-radius:8px;cursor:pointer;background:#2C5F2D;color:#fff}
 button:hover{background:#3d7a3e}
 button.skip{background:#5a5a5a}
 button.done{background:#C9A227;color:#1b1b1b;font-weight:600}
 .hint{font-size:12.5px;color:#9db98c;margin:10px 0 4px;text-align:center;line-height:1.6}
 textarea{width:min(90vw,760px);height:180px;margin-top:10px;font-family:ui-monospace,monospace;font-size:12px}
</style></head><body>
<header>
 <h1>Severity grading &mdash; __FOLDER__</h1>
 <div id="bar"><div id="fill"></div></div>
 <div id="meta"></div>
</header>
<img id="pic" alt="(image failed to load)" onerror="document.getElementById('meta').textContent='Could not load '+IMAGES[i]+' - is the images folder next to this file?'">
<div class="hint">
 How much of the <b>LEAF</b> shows disease symptoms? Ignore the background.<br>
 <b>0</b> none &nbsp;|&nbsp; <b>1</b> mild, under 5% &nbsp;|&nbsp; <b>2</b> moderate, 5&ndash;20% &nbsp;|&nbsp;
 <b>3</b> severe, over 20% &nbsp;|&nbsp; <b>S</b> skip if unsure
</div>
<div class="btns">
 <button onclick="grade(0)">0 &middot; none</button>
 <button onclick="grade(1)">1 &middot; mild</button>
 <button onclick="grade(2)">2 &middot; moderate</button>
 <button onclick="grade(3)">3 &middot; severe</button>
 <button class="skip" onclick="grade('')">S &middot; skip</button>
 <button class="skip" onclick="back()">&larr; undo</button>
</div>
<div class="btns" style="margin-top:12px">
 <button class="done" onclick="finish()">Show my grades.csv</button>
</div>
<textarea id="out" style="display:none" readonly></textarea>
<script>
const IMAGES=__IMAGES__, KEY="grades___FOLDER__";
let g=JSON.parse(localStorage.getItem(KEY)||"{}"), i=0;
function firstUngraded(){for(let k=0;k<IMAGES.length;k++)if(!(IMAGES[k] in g))return k;return IMAGES.length;}
function show(){
  i=Math.min(i,IMAGES.length-1);
  const done=Object.keys(g).length;
  document.getElementById("fill").style.width=(100*done/IMAGES.length)+"%";
  document.getElementById("meta").textContent=`${done} / ${IMAGES.length} graded  -  now showing #${i+1}: ${IMAGES[i]}`;
  document.getElementById("pic").src="images/"+encodeURIComponent(IMAGES[i]);
}
function grade(v){g[IMAGES[i]]=v;localStorage.setItem(KEY,JSON.stringify(g));i=firstUngraded();
  if(i>=IMAGES.length){finish();}else{show();}}
function back(){i=Math.max(0,i-1);delete g[IMAGES[i]];localStorage.setItem(KEY,JSON.stringify(g));show();}
function finish(){
  let csv="image,manual_grade\\n";
  IMAGES.forEach(n=>{csv+=n+","+(n in g?g[n]:"")+"\\n";});
  const t=document.getElementById("out");t.style.display="block";t.value=csv;t.select();
  document.getElementById("meta").textContent="Done. Copy the text below into grades.csv, or save it as grades.csv.";
}
document.addEventListener("keydown",e=>{
  if(["0","1","2","3"].includes(e.key))grade(parseInt(e.key));
  else if(e.key.toLowerCase()==="s")grade("");
  else if(e.key==="Backspace")back();});
i=firstUngraded();show();
</script></body></html>
"""

INSTRUCTIONS = """Severity grading - COMP9444 Project 090
=======================================

EASIEST WAY: double-click `grade.html`. It shows one leaf at a time and you
press 0-3. Nothing to install, progress is saved automatically.

You are grading how much of the LEAF AREA shows disease symptoms (lesions,
spots, discolouration, necrosis).

The file names are dataset ids - they are meant to look meaningless, and the
disease name is deliberately hidden so your judgement is not biased. Just look
at the leaf.

If you prefer the spreadsheet, write one number in the `manual_grade` column:

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
- The rows are in no particular order; grade each image on its own.

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
        names = []
        with (folder / "grades.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["no", "image", "manual_grade"])
            for row in shard:
                source = Path(row["path"].replace("\\", "/"))
                name = f"{source.stem}.jpg"
                shutil.copy2(source, folder / "images" / name)
                writer.writerow([len(names) + 1, name, ""])
                names.append(name)
        (folder / "grade.html").write_text(
            GRADER_HTML.replace("__IMAGES__", json.dumps(names))
                       .replace("__FOLDER__", folder.name), encoding="utf-8")
        print(f"  {folder.as_posix()}: {len(shard)} images (open grade.html)")
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
        key = f"{Path(row['path'].replace(chr(92), '/')).stem}.jpg"
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
