#!/usr/bin/env python3
"""Evaluate a YOLO defect model on a validation dataset and print metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO defect detection model and show precision, recall, mAP, and confusion matrix."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("yolov26n.pt"),
        help="Path to the YOLO model checkpoint.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data_pseudo.yaml"),
        help="YOLO dataset config file for validation.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for validation.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for validation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device (cpu, cuda:0, auto).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold used during validation.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="NMS IoU threshold used during validation.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("runs/val"),
        help="Directory where evaluation artifacts are saved.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save confusion matrix plot and optional visualizations.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the confusion matrix plot.",
    )
    return parser.parse_args()


def format_row(values: list[Any], widths: list[int]) -> str:
    return " | ".join(str(value).ljust(width) for value, width in zip(values, widths))


def format_table(headers: list[str], rows: list[list[Any]]) -> str:
    columns = list(zip(headers, *rows))
    widths = [max(len(str(value)) for value in col) for col in columns]
    lines = [format_row(headers, widths), format_row(["-" * width for width in widths], widths)]
    for row in rows:
        lines.append(format_row(row, widths))
    return "\n".join(lines)


def print_metrics(results: Any) -> None:
    box = results.box
    names = getattr(results, "names", {}) or {}
    class_names = [names.get(i, str(i)) for i in range(box.nc)]

    print("\n=== Validation Metrics ===")
    print(f"Precision (mean): {box.mp:.4f}")
    print(f"Recall    (mean): {box.mr:.4f}")
    print(f"mAP@0.5   : {box.map50:.4f}")
    print(f"mAP@0.5:0.95: {box.map:.4f}\n")

    rows = []
    for i, label in enumerate(class_names):
        p, r, ap50, ap = box.class_result(i)
        rows.append([label, f"{p:.4f}", f"{r:.4f}", f"{ap50:.4f}", f"{ap:.4f}"])

    print(format_table(["Class", "Precision", "Recall", "AP50", "AP"], rows))


def print_confusion_matrix(confusion_matrix: Any) -> None:
    matrix = confusion_matrix.matrix
    names = list(confusion_matrix.names.values()) if hasattr(confusion_matrix.names, "values") else []
    if confusion_matrix.task == "detect":
        names = [*names, "background"]

    rows = []
    for i, row in enumerate(matrix.tolist()):
        rows.append([names[i], *[int(v) for v in row]])

    headers = ["predicted / actual", *names]
    print("\n=== Confusion Matrix ===")
    print("(rows = predicted, columns = actual)")
    print(format_table(headers, rows))


def validate_model(args: argparse.Namespace) -> tuple[Any, Any]:
    if not args.weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {args.weights}\n"
            f"Train a model first with:\n"
            f"  python train.py --data {args.data} --model yolov26n.pt --project runs/train --name ceramic_defects\n"
            f"Or evaluate a pretrained checkpoint directly, for example: yolov26n.pt"
        )
    if not args.data.exists():
        raise FileNotFoundError(
            f"Data config not found: {args.data}\n"
            f"Create or verify the dataset config before evaluation."
        )

    model = YOLO(str(args.weights))
    custom = {
        "data": str(args.data),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "plots": True,
        "visualize": False,
        "save_txt": False,
        "save_json": False,
        "save_conf": False,
        "task": "detect",
        "mode": "val",
    }
    args_dict = {**model.overrides, **custom}

    validator_cls = model._smart_load("validator")
    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    validator = validator_cls(args=args_dict, save_dir=save_dir, _callbacks=model.callbacks)
    validator(model=model.model)
    return validator.metrics, validator.confusion_matrix


def main() -> int:
    args = parse_args()
    try:
        metrics, confusion = validate_model(args)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print_metrics(metrics)
    print_confusion_matrix(confusion)

    if args.plots:
        try:
            confusion.plot(normalize=args.normalize, save_dir=str(args.save_dir))
            print(f"Saved confusion matrix plot to {args.save_dir}")
        except Exception as exc:
            print(f"Warning: could not save confusion plot: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
