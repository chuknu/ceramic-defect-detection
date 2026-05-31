#!/usr/bin/env python3
"""Capture images from a camera or video source for dataset collection."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture images from a camera or video file for dataset creation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/raw_images"),
        help="Directory to save captured images.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index to use when capturing from a webcam.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional local video file to capture frames from instead of a camera.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Time in seconds between automatic captures when in auto mode.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Stop after saving this many images (0 means unlimited).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically capture images every interval seconds.",
    )
    return parser.parse_args()


def create_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def next_filename(output_dir: Path) -> Path:
    existing = sorted(output_dir.glob("*.jpg"))
    if not existing:
        return output_dir / "img_000001.jpg"
    last = existing[-1].stem
    try:
        number = int(last.split("_")[-1])
    except ValueError:
        number = len(existing)
    return output_dir / f"img_{number + 1:06d}.jpg"


def open_video_source(camera_index: int, video: Path | None) -> cv2.VideoCapture:
    if video is not None:
        if not video.exists():
            raise FileNotFoundError(f"Video file not found: {video}")
        return cv2.VideoCapture(str(video))
    return cv2.VideoCapture(camera_index)


def main() -> int:
    args = parse_args()
    output_dir = create_output_dir(args.output_dir)
    capture = open_video_source(args.camera_index, args.video)
    if not capture.isOpened():
        print("Error: unable to open the camera or video source.")
        return 1

    print("Press SPACE to save a frame, q to quit.")
    if args.auto:
        print(f"Auto-capture enabled: saving one image every {args.interval:.1f} seconds.")

    window_name = "Capture Images"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    last_capture = time.time()
    saved = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Stream ended or camera disconnected.")
                break

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            now = time.time()

            if args.auto and (now - last_capture) >= args.interval:
                filename = next_filename(output_dir)
                cv2.imwrite(str(filename), frame)
                saved += 1
                last_capture = now
                print(f"Saved {filename}")

            if key == ord(" "):
                filename = next_filename(output_dir)
                cv2.imwrite(str(filename), frame)
                saved += 1
                last_capture = now
                print(f"Saved {filename}")

            if key == ord("q") or key == 27:
                break

            if args.max_images > 0 and saved >= args.max_images:
                print(f"Captured {saved} images, stopping.")
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Finished capturing {saved} images to {output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
