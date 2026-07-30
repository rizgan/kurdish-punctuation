#!/usr/bin/env python3
"""Prepare train/validation/test JSONL from Wikipedia article JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_punctuation.dataset import build_processed_dataset  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Raw JSONL with article_id+text")
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ds = cfg.get("dataset", {})
    stats = build_processed_dataset(
        args.input,
        args.output_dir,
        seed=int(cfg.get("project", {}).get("seed", 42)),
        map_ellipsis_to_period=bool(ds.get("map_ellipsis_to_period", True)),
        min_words=int(ds.get("min_words_per_sample", 3)),
        max_words=int(ds.get("max_words_per_sample", 180)),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
