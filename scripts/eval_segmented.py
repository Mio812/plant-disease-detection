"""Re-score the trained models on the background-removed `segmented` variant.

Same leaves, same test split - only the background is masked out. The accuracy
drop measures how much each model relied on background/capture cues.

Usage:
    python -m scripts.eval_segmented --config configs/default.yaml
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from src.bias import name_key, segmented_index
from src.config import Config
from src.data import build_transforms
from src.ensemble import combine
from src.models import build_model
from src.utils import get_device, load_checkpoint, set_seed

MEMBERS = ["custom_cnn", "resnet18", "mobilenet_v2"]


class PathDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on background-removed images.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", type=float, nargs=3, default=None,
                        help="Ensemble weights; defaults to equal.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    device = get_device()
    out_dir = Path(cfg.output_dir)

    base = ImageFolder(cfg.data.root)
    segmented_root = str(Path(cfg.data.root).parent / "segmented")
    n = len(base)
    n_test = int(n * cfg.data.test_split)
    test_idx = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed)).tolist()[:n_test]

    index = {c: segmented_index(segmented_root, c) for c in base.classes}
    paths, labels = [], []
    for i in test_idx:
        colour_path, label = base.samples[i]
        match = index[base.classes[label]].get(name_key(Path(colour_path).name))
        if match:
            paths.append(match)
            labels.append(label)
    print(f"matched {len(paths)}/{len(test_idx)} test images in the segmented variant")

    loader = DataLoader(PathDataset(paths, labels, build_transforms(cfg.data.image_size, False)),
                        batch_size=cfg.data.batch_size, shuffle=False,
                        num_workers=cfg.data.num_workers)
    targets = torch.tensor(labels)

    probs, results = [], {}
    for name in MEMBERS:
        model = build_model(name, len(base.classes), pretrained=False).to(device)
        model.load_state_dict(load_checkpoint(out_dir / f"{name}_best.pth", map_location=device)["model"])
        model.eval()
        with torch.no_grad():
            member = torch.cat([torch.softmax(model(x.to(device)), 1).cpu() for x, _ in loader])
        probs.append(member)
        results[name] = float((member.argmax(1) == targets).float().mean() * 100)
        print(f"  {name:14s} segmented accuracy {results[name]:.2f}%")

    results["ensemble"] = float((combine(probs, args.weights).argmax(1) == targets).float().mean() * 100)
    print(f"  {'ensemble':14s} segmented accuracy {results['ensemble']:.2f}%")
    (out_dir / "segmented_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
