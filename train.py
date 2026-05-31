#!/usr/bin/env python3
"""Train or fine-tune a YOLO defect detection model from a dataset config."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO defect detection model using a YOLO dataset config."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data_pseudo.yaml"),
        help="YOLO data config file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov26n.pt",
        help="Starting YOLO model checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device for training (cpu, cuda:0, auto).",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/train"),
        help="Project folder to save training results.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ceramic_defects",
        help="Experiment name under the project folder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data.exists():
        print(f"Error: data config not found: {args.data}")
        return 1

    model = YOLO(str(args.model))
    print(f"Training model: {args.model}")
    print(f"Data config: {args.data}")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
    )
    print("Training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
