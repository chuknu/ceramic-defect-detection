#!/usr/bin/env python3
"""Run defect detection on images, video files, or USB camera sources."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2

from defect_model import DefectModel

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run defect detection on an image, video file, folder of images, or USB camera source."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Path to an image, video file, or directory of images. Leave empty to use the USB camera.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov26n.pt",
        help="Path to a defect detection YOLO model file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Directory to save annotated frames and logs.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="USB camera index when using live capture.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for defect detection.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def find_images(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and is_image_file(path)
    )


def load_image(source: Path) -> any:
    image = cv2.imread(str(source))
    if image is None:
        raise FileNotFoundError(f"Could not read image file: {source}")
    return image


def open_capture(source: str | None, camera_index: int) -> cv2.VideoCapture:
    if source is None:
        return cv2.VideoCapture(camera_index)
    candidate = Path(source)
    if candidate.exists() and not is_image_file(candidate):
        return cv2.VideoCapture(str(candidate))
    raise FileNotFoundError(f"Video source not found or unsupported: {source}")


def save_annotation(frame: any, output_dir: Path, prefix: str, frame_index: int) -> Path:
    path = output_dir / f"{prefix}_{frame_index:04d}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Using model: {args.model}")
    print(f"Output directory: {args.output}")

    try:
        detector = DefectModel(model_path=args.model, conf=args.conf, device="auto")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    source_path = Path(args.source) if args.source else None
    if source_path is not None and source_path.exists() and source_path.is_dir():
        image_paths = find_images(source_path)
        if not image_paths:
            print(f"No supported image files found in directory: {source_path}")
            return 1

        output_files = []
        total_defects = 0
        print(f"Running batch inference on {len(image_paths)} images from {source_path}")

        for index, image_path in enumerate(image_paths, start=1):
            frame = load_image(image_path)
            annotated, defects = detector.predict(frame)
            relative_path = image_path.relative_to(source_path)
            output_path = args.output / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)
            defect_count = len(defects)
            total_defects += defect_count
            output_files.append(output_path)
            print(f"[{index}/{len(image_paths)}] {relative_path}: {defect_count} defects -> {output_path}")

        print(f"Batch inference complete. Saved {len(output_files)} annotated images to {args.output}")
        print(f"Total defects detected: {total_defects}")
        return 0

    if args.source is None or (source_path is not None and source_path.exists() and not is_image_file(source_path)):
        capture = open_capture(args.source, args.camera_index)
        if not capture.isOpened():
            print("Error: unable to open the camera or video source.")
            return 1

        frame_count = 0
        total_defects = 0
        print("Starting live detection. Press Ctrl+C to stop.")

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    print("Stream ended or camera disconnected.")
                    break

                annotated, defects = detector.predict(frame)
                defect_count = len(defects)
                total_defects += defect_count
                annotated_path = save_annotation(annotated, args.output, "annotated", frame_count)

                print(
                    f"Frame {frame_count}: {defect_count} defects, "
                    f"saved annotated frame to {annotated_path}"
                )
                frame_count += 1
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("Stopping detection.")
        finally:
            capture.release()

        print(f"Processed {frame_count} frames with {total_defects} total defects detected.")
        return 0

    image_source = Path(args.source)
    if image_source.exists() and is_image_file(image_source):
        frame = load_image(image_source)
        annotated, defects = detector.predict(frame)
        defect_count = len(defects)
        annotated_path = save_annotation(annotated, args.output, "image", 0)

        print(f"Detected {defect_count} defects in {image_source}")
        print(f"Saved annotated image to {annotated_path}")
        return 0

    print("Error: invalid source. Use a valid image file, video file, or omit the source to use the camera.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
