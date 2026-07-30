#!/usr/bin/env python3
"""Download Kurmanji Wikipedia articles dump (kuwiki)."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://dumps.wikimedia.org/kuwiki/latest/kuwiki-latest-pages-articles.xml.bz2"
)
# Mirror example (often faster / higher limits):
# https://dumps.wikimedia.your.org/... — see https://dumps.wikimedia.org/mirrors.html
USER_AGENT = "kurdish-punctuation/0.1 (research fine-tune; local RTX4090)"


def download(url: str, dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"Already exists: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading:\n  {url}\n-> {dest}")

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        total = resp.headers.get("Content-Length")
        total_i = int(total) if total else None
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total_i:
                pct = 100.0 * done / total_i
                print(f"\r  {done / 1e6:.1f}/{total_i / 1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
            else:
                print(f"\r  {done / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=DEFAULT_URL, help="Dump URL (default: kuwiki latest articles)")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/kuwiki-latest-pages-articles.xml.bz2"),
        help="Output path",
    )
    p.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = p.parse_args()
    try:
        download(args.url, args.out, force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Tip: if dumps.wikimedia.org rate-limits, pick a mirror from "
            "https://dumps.wikimedia.org/mirrors.html",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
