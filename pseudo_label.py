#!/usr/bin/env python3
"""Generate YOLO pseudo-labels from unlabeled images using the current defect detection model."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Any

import cv2

from defect_model import DefectModel

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create pseudo-labeled YOLO files from a folder of unlabeled images."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing unlabeled images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/pseudo"),
        help="Output dataset base directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov26n.pt",
        help="Path to the defect YOLO model to use for pseudo-labeling.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train/validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset split.",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Save annotated images for review.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages.",
    )
    return parser.parse_args()


def find_images(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()
    )


def create_dirs(base_dir: Path, save_annotated: bool) -> dict[str, Path]:
    dirs = {
        "train_images": base_dir / "images" / "train",
        "val_images": base_dir / "images" / "val",
        "train_labels": base_dir / "labels" / "train",
        "val_labels": base_dir / "labels" / "val",
    }
    if save_annotated:
        dirs["annotated"] = base_dir / "annotated"
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def xyxy_to_yolo(xyxy: list[float], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    x_center = (x1 + x2) / 2.0 / image_width
    y_center = (y1 + y2) / 2.0 / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return x_center, y_center, width, height


def write_label_file(label_path: Path, labels: list[str]) -> None:
    label_path.write_text("\n".join(labels))


def save_image(source: Path, dest: Path) -> None:
    if source.resolve() == dest.resolve():
        return
    shutil.copy2(str(source), str(dest))


def process_image(
    model: DefectModel,
    image_path: Path,
    label_dest: Path,
    save_annotated: bool,
    annotated_path: Path | None = None,
) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    annotated, defects = model.predict(image)
    labels: list[str] = []
    for defect in defects:
        bbox = defect["bbox"]
        class_id = int(defect["label"].split("_")[-1])
        x_center, y_center, width, height = xyxy_to_yolo(bbox, image.shape[1], image.shape[0])
        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    write_label_file(label_dest, labels)
    if save_annotated and annotated_path is not None:
        cv2.imwrite(str(annotated_path), annotated)
    return len(labels)


def main() -> int:
    args = parse_args()
    if not args.source_dir.exists() or not args.source_dir.is_dir():
        print(f"Error: source directory not found: {args.source_dir}")
        return 1

    image_paths = find_images(args.source_dir)
    if not image_paths:
        print(f"No supported images found in {args.source_dir}")
        return 1

    random.seed(args.seed)
    random.shuffle(image_paths)
    dirs = create_dirs(args.output_dir, args.save_annotated)

    model = DefectModel(model_path=args.model, conf=args.conf, device="auto")
    split_index = int(len(image_paths) * args.split)

    count = 0
    for index, image_path in enumerate(image_paths):
        subset = "train" if index < split_index else "val"
        image_dest = dirs[f"{subset}_images"] / image_path.name
        label_dest = dirs[f"{subset}_labels"] / f"{image_path.stem}.txt"
        annotated_dest = None
        if args.save_annotated:
            annotated_dest = dirs["annotated"] / image_path.name

        save_image(image_path, image_dest)
        defects_written = process_image(model, image_dest, label_dest, args.save_annotated, annotated_dest)
        count += 1
        if args.verbose:
            print(
                f"[{subset}] {image_path.name}: {defects_written} pseudo labels -> {label_dest}"
            )

    print(f"Generated {count} pseudo-labeled images in {args.output_dir}")
    print("Review the generated labels under labels/train and labels/val, then correct any mistakes before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
