#!/usr/bin/env python3
"""Download permissively-licensed images from Wikimedia Commons for given search terms.

This script only downloads files whose license metadata indicates Public Domain or CC0.
It saves images under `datasets/web_images/<slug>/` and writes a JSONL metadata file for each image.

Usage:
  python scrape_wikimedia.py --queries "ceramic tile crack,ceramic chip" --per-query 10
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import quote_plus

import requests


WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "ceramic-defect-bot/0.1 (https://example.com)"}
VIDEO_EXTENSIONS = {"webm", "ogv", "ogg", "mp4", "mov", "mkv"}
TILE_KEYWORDS = {"tile", "tiles", "floor tile", "wall tile", "ceramic tile", "glazed tile", "tile crack", "tile chip", "ceramic tile crack", "ceramic tile chip"}


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape Wikimedia Commons for permissively licensed tiles and media")
    parser.add_argument("--queries", type=str, required=True, help="Comma-separated search queries")
    parser.add_argument("--per-query", type=int, default=10, help="Max media items per query")
    parser.add_argument("--output", type=Path, default=Path("datasets/tile_images"), help="Output base dir")
    parser.add_argument("--allow-cc", action="store_true", help="Also allow Creative Commons licensed images (CC-BY, CC-BY-SA). Attribution metadata will be saved.")
    parser.add_argument("--include-videos", action="store_true", help="Download video files in addition to images.")
    return parser.parse_args()


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def is_permissive_license(extmeta: dict, allow_cc: bool = False) -> bool:
    # Look for public domain or CC0 indicators in extmetadata fields
    for key in ("LicenseShortName", "License", "LicenseUrl"):
        v = extmeta.get(key)
        if not v:
            continue
        text = v.get("value") if isinstance(v, dict) else str(v)
        if not text:
            continue
        t = text.lower()
        if "public domain" in t or "cc0" in t or "cc0 1.0" in t:
            return True
        if allow_cc and ("creative commons" in t or "cc-by" in t or "cc-by-sa" in t):
            return True
    return False


def title_matches_tile_keywords(title: str) -> bool:
    if not title:
        return False
    lower = title.lower()
    return any(keyword in lower for keyword in TILE_KEYWORDS)


def is_video_url(url: str) -> bool:
    return any(url.lower().endswith(f".{ext}") for ext in VIDEO_EXTENSIONS)


def search_images(query: str, limit: int = 10) -> list[dict]:
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "iiprop": "url|extmetadata|mime",
        "iiurlwidth": "1024",
    }
    resp = requests.get(WIKIMEDIA_API, params=params, timeout=15, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    results = []
    for pid, page in pages.items():
        imageinfo = page.get("imageinfo")
        if not imageinfo:
            continue
        info = imageinfo[0]
        url = info.get("thumburl") or info.get("url")
        extmeta = info.get("extmetadata", {})
        mime = info.get("mime", "")
        results.append({"title": page.get("title"), "url": url, "extmeta": extmeta, "mime": mime})
    return results


def download_file(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=30, headers=HEADERS)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    base = args.output
    base.mkdir(parents=True, exist_ok=True)

    total = 0
    for q in queries:
        print(f"Searching: {q}")
        results = search_images(q, limit=args.per_query * 2)
        slug = slugify(q)
        out_dir = base / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / "metadata.jsonl"
        saved = 0
        for item in results:
            if saved >= args.per_query:
                break
            extmeta = item.get("extmeta", {})
            if not is_permissive_license(extmeta, allow_cc=args.allow_cc):
                continue
            title = item.get("title", "")
            if not title_matches_tile_keywords(title):
                continue
            url = item.get("url")
            if not url:
                continue

            mime = item.get("mime", "")
            if mime.startswith("application/pdf"):
                continue
            is_video = is_video_url(url) or mime.startswith("video")
            if is_video and not args.include_videos:
                continue
            if not is_video and mime.startswith("video") and not args.include_videos:
                continue

            fname = Path(url.split("/")[-1].split(":")[-1])
            dest = out_dir / fname
            ok = download_file(url, dest)
            if not ok:
                continue
            meta = {
                "title": title,
                "url": url,
                "file": str(dest),
                "mime": mime,
                "extmeta": {k: (v.get("value") if isinstance(v, dict) else v) for k, v in extmeta.items()},
            }
            with meta_path.open("a") as fh:
                fh.write(json.dumps(meta) + "\n")
            saved += 1
            total += 1
            media_type = "video" if is_video else "image"
            print(f"Saved {media_type}: {dest}")

        print(f"Saved {saved} permissively-licensed tiles for query '{q}' to {out_dir}")

    print(f"Done. Total saved: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
