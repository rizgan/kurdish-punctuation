#!/usr/bin/env python3
"""Prepare continuous word-window train/validation/test JSONL from Wikipedia articles."""

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
    p.add_argument("--max-articles", type=int, default=None, help="Optional subset for sanity runs")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ds = cfg.get("dataset", {})
    model_name = cfg.get("model", {}).get("name", "FacebookAI/xlm-roberta-base")

    stats = build_processed_dataset(
        args.input,
        args.output_dir,
        seed=int(ds.get("seed", cfg.get("project", {}).get("seed", 42))),
        map_ellipsis_to_period=bool(ds.get("map_ellipsis_to_period", True)),
        tokenizer_name=model_name,
        max_model_length=int(ds.get("max_model_length", cfg.get("model", {}).get("max_length", 256))),
        min_words_per_window=int(ds.get("min_words_per_window", 80)),
        target_words_min=int(ds.get("target_words_min", 110)),
        target_words_max=int(ds.get("target_words_max", 180)),
        train_overlap_words=int(ds.get("train_overlap_words", 8)),
        val_overlap_words=int(ds.get("val_overlap_words", 0)),
        test_overlap_words=int(ds.get("test_overlap_words", 0)),
        max_articles=args.max_articles,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    frac = stats["period_window_stats"]["train"]["fraction_of_period_labels_at_final_word"]
    print(f"train fraction_of_period_labels_at_final_word={frac:.4f}")
    if frac > 0.15:
        print("WARNING: too many PERIOD labels sit on the final word of windows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
