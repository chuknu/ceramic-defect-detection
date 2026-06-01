#!/usr/bin/env python3
"""Download images from a list of URLs into a directory.

Usage:
  python download_images.py --urls urls.txt --output datasets/raw_images

`urls.txt` should contain one image URL per line.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Download images from a list of URLs.")
    parser.add_argument("--urls", type=Path, required=True, help="Text file with one image URL per line")
    parser.add_argument("--output", type=Path, default=Path("datasets/raw_images"), help="Output directory")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds")
    return parser.parse_args()


def download(url: str, dest: Path, timeout: float = 10.0) -> bool:
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            return False
        ext = content_type.split("/")[-1].split(";")[0]
        if ext.lower() not in {"jpeg", "jpg", "png", "bmp", "tiff", "webp"}:
            ext = "jpg"
        dest.write_bytes(resp.content)
        return True
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    urls = [line.strip() for line in args.urls.read_text().splitlines() if line.strip()]
    count = 0
    for i, url in enumerate(urls, start=1):
        filename = args.output / f"img_{i:06d}.jpg"
        ok = download(url, filename, timeout=args.timeout)
        if ok:
            count += 1
            print(f"Saved {filename}")
        else:
            print(f"Failed: {url}")

    print(f"Downloaded {count}/{len(urls)} images to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
