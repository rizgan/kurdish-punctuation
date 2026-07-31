#!/usr/bin/env python3
"""Prepare continuous-window capitalization train/validation/test JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_capitalization.dataset import build_processed_dataset  # noqa: E402
from kurmanji_capitalization.sentence_rule import SentenceRuleConfig  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Raw JSONL with article_id+text")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--config", type=Path, default=Path("configs/capitalization-v1.yaml"))
    p.add_argument("--max-articles", type=int, default=None)
    p.add_argument("--max-train-words", type=int, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ds = cfg.get("dataset", {})
    paths = cfg.get("paths", {})
    model_name = cfg.get("model", {}).get("name", "FacebookAI/xlm-roberta-base")
    output_dir = args.output_dir or Path(paths.get("processed_dir", "data/processed_capitalization"))

    sr = cfg.get("sentence_rule", {})
    sentence_cfg = SentenceRuleConfig(
        capitalize_first_word=bool(sr.get("capitalize_first_word", True)),
        capitalize_after_period=bool(sr.get("capitalize_after_period", True)),
        capitalize_after_question=bool(sr.get("capitalize_after_question", True)),
        capitalize_after_exclamation=bool(sr.get("capitalize_after_exclamation", True)),
        skip_leading_quotes=bool(sr.get("skip_leading_quotes", True)),
        capitalize_word_after_leading_number=bool(
            sr.get("capitalize_word_after_leading_number", False)
        ),
    )

    max_articles = args.max_articles
    if max_articles is None and ds.get("max_articles") is not None:
        max_articles = int(ds["max_articles"])
    max_train_words = args.max_train_words
    if max_train_words is None and ds.get("max_train_words") is not None:
        max_train_words = int(ds["max_train_words"])

    stats = build_processed_dataset(
        args.input,
        output_dir,
        seed=int(ds.get("seed", cfg.get("project", {}).get("seed", 42))),
        map_ellipsis_to_period=bool(ds.get("map_ellipsis_to_period", True)),
        tokenizer_name=model_name,
        max_model_length=int(ds.get("max_model_length", cfg.get("model", {}).get("max_length", 256))),
        min_tokens_per_window=int(ds.get("min_tokens_per_window", 80)),
        target_tokens_min=int(ds.get("target_tokens_min", 110)),
        target_tokens_max=int(ds.get("target_tokens_max", 180)),
        train_overlap_tokens=int(ds.get("train_overlap_tokens", 8)),
        val_overlap_tokens=int(ds.get("val_overlap_tokens", 0)),
        test_overlap_tokens=int(ds.get("test_overlap_tokens", 0)),
        max_articles=max_articles,
        max_train_words=max_train_words,
        sentence_cfg=sentence_cfg,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
