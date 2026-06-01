#!/usr/bin/env python3
"""Prepare a YOLO-style dataset from raw images and label sources.

It copies images into `output/images/{train,val}` and labels into `output/labels/{train,val}`
and writes a `data_pseudo.yaml` referencing the created paths and class names if provided.

Usage example:
  python prepare_dataset.py --images datasets/raw_images --labels datasets/hitl/labels --output datasets/pseudo --split 0.8
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset structure from images and labels")
    parser.add_argument("--images", type=Path, required=True, help="Directory with raw images")
    parser.add_argument("--labels", type=Path, default=Path(""), help="Directory with label .txt files (optional)")
    parser.add_argument("--output", type=Path, default=Path("datasets/pseudo"), help="Output dataset base directory")
    parser.add_argument("--split", type=float, default=0.8, help="Train fraction (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--names", type=Path, default=Path("data_pseudo.yaml"), help="Existing YAML file to copy names from (optional)")
    return parser.parse_args()


def find_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts and p.is_file())


def create_dirs(output: Path):
    dirs = {
        "train_images": output / "images" / "train",
        "val_images": output / "images" / "val",
        "train_labels": output / "labels" / "train",
        "val_labels": output / "labels" / "val",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def copy_files(image_paths: List[Path], labels_dir: Path, dirs: Dict[str, Path], split: float, seed: int):
    random.seed(seed)
    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * split)
    for idx, img_path in enumerate(image_paths):
        subset = "train" if idx < split_idx else "val"
        dest_img = (dirs[f"{subset}_images"] / img_path.name)
        shutil.copy2(str(img_path), str(dest_img))
        label_src = labels_dir / f"{img_path.stem}.txt" if labels_dir and labels_dir.exists() else None
        if label_src and label_src.exists():
            dest_lbl = dirs[f"{subset}_labels"] / f"{img_path.stem}.txt"
            shutil.copy2(str(label_src), str(dest_lbl))


def read_names_from_yaml(yaml_path: Path) -> Dict[int, str]:
    if not yaml_path.exists():
        return {}
    names: Dict[int, str] = {}
    in_names = False
    for line in yaml_path.read_text().splitlines():
        if not in_names:
            if line.strip().startswith("names:"):
                in_names = True
            continue
        if not line.startswith((" ", "\t")):
            break
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        if k.strip().isdigit():
            names[int(k.strip())] = v.strip()
    return names


def write_data_yaml(output: Path, names: Dict[int, str]):
    yaml_path = output.parent / "data_pseudo.yaml"
    lines = [f"train: {output}/images/train", f"val: {output}/images/val", "names:"]
    if names:
        for k in sorted(names.keys()):
            lines.append(f"  {k}: {names[k]}")
    else:
        lines.append("  0: defect")
    yaml_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote dataset yaml to {yaml_path}")


def main() -> int:
    args = parse_args()
    if not args.images.exists():
        print(f"Images directory not found: {args.images}")
        return 1
    image_paths = find_images(args.images)
    if not image_paths:
        print("No images found.")
        return 1

    dirs = create_dirs(args.output)
    copy_files(image_paths, args.labels, dirs, args.split, args.seed)
    names = read_names_from_yaml(args.names) if args.names else {}
    write_data_yaml(args.output, names)
    print(f"Prepared dataset at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
