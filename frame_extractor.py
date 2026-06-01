#!/usr/bin/env python3
"""Extract frames from video files at a fixed interval and optionally skip near-duplicate frames.

Usage:
    python frame_extractor.py --input video.mp4 --output datasets/raw_images --interval 1.0 --dedupe 0.9

"""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("--input", type=Path, required=True, help="Path to video file")
    parser.add_argument("--output", type=Path, default=Path("datasets/raw_images"), help="Output directory")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between saved frames")
    parser.add_argument(
        "--dedupe",
        type=float,
        default=0.0,
        help="Histogram correlation threshold (0-1). If >0, skip frames with correlation >= dedupe.",
    )
    return parser.parse_args()


def hist_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    # compute grayscale histogram correlation
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h1 = cv2.calcHist([g1], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([g2], [0], None, [256], [0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"Video not found: {args.input}")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.input))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * args.interval)))

    frame_idx = 0
    saved = 0
    last_saved = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            save = True
            if args.dedupe > 0 and last_saved is not None:
                corr = hist_correlation(frame, last_saved)
                if corr >= args.dedupe:
                    save = False

            if save:
                out_path = args.output / f"frame_{saved:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                last_saved = frame.copy()
                saved += 1
                print(f"Saved {out_path}")

        frame_idx += 1

    cap.release()
    print(f"Finished. Saved {saved} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
