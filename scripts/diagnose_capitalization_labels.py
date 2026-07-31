#!/usr/bin/env python3
"""Diagnose capitalization auto-labels before full training.

Samples TITLE/UPPER examples, estimates wiki-header noise, and writes a review report.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_capitalization.casing import kurmanji_lower  # noqa: E402
from kurmanji_capitalization.dataset import (  # noqa: E402
    load_raw_jsonl,
    tokens_and_labels_from_article,
)
from kurmanji_capitalization.normalization import normalize_for_dataset  # noqa: E402
from kurmanji_capitalization.sentence_rule import SentenceRuleConfig  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("data/raw/wikipedia.jsonl"))
    p.add_argument("--config", type=Path, default=Path("configs/capitalization-diagnostic.yaml"))
    p.add_argument("--max-articles", type=int, default=500)
    p.add_argument("--examples-per-label", type=int, default=40)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed_capitalization_diagnostic/label_diagnosis.json"),
    )
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sr = cfg.get("sentence_rule", {})
    sentence_cfg = SentenceRuleConfig(**{k: sr[k] for k in SentenceRuleConfig.__dataclass_fields__ if k in sr})
    seed = int(cfg.get("project", {}).get("seed", 42))
    rng = random.Random(seed)

    articles = load_raw_jsonl(args.input)
    rng.shuffle(articles)
    articles = articles[: args.max_articles]

    label_counts: Counter[str] = Counter()
    title_examples: list[dict] = []
    upper_examples: list[dict] = []
    keep_as_title_risk: list[dict] = []
    short_title_tokens: Counter[str] = Counter()
    n_ok = 0
    n_skip = 0
    total_words = 0

    for art in articles:
        text = normalize_for_dataset(art["text"])
        triple = tokens_and_labels_from_article(text, sentence_cfg=sentence_cfg)
        if not triple:
            n_skip += 1
            continue
        tokens, labels, originals = triple
        n_ok += 1
        for tok, lab, orig in zip(tokens, labels, originals):
            label_counts[lab] += 1
            if lab != "IGNORE" and tok not in ".,?!":
                total_words += 1
            if lab == "TITLE":
                short_title_tokens[kurmanji_lower(orig)] += 1
                if len(title_examples) < args.examples_per_label * 3:
                    title_examples.append(
                        {
                            "article_id": art["article_id"],
                            "input": tok,
                            "original": orig,
                            "label": lab,
                        }
                    )
            elif lab == "UPPER":
                if len(upper_examples) < args.examples_per_label * 3:
                    upper_examples.append(
                        {
                            "article_id": art["article_id"],
                            "input": tok,
                            "original": orig,
                            "label": lab,
                        }
                    )

    # Heuristic: very frequent TITLE tokens may be wiki section/style noise
    frequent_titles = [
        {"token": t, "count": c}
        for t, c in short_title_tokens.most_common(30)
        if c >= 5
    ]

    rng.shuffle(title_examples)
    rng.shuffle(upper_examples)
    report = {
        "articles_sampled": len(articles),
        "articles_labeled": n_ok,
        "articles_skipped": n_skip,
        "approx_words": total_words,
        "label_counts": dict(label_counts),
        "title_share_of_trainable": (
            label_counts["TITLE"]
            / max(1, label_counts["KEEP"] + label_counts["TITLE"] + label_counts["UPPER"])
        ),
        "upper_share_of_trainable": (
            label_counts["UPPER"]
            / max(1, label_counts["KEEP"] + label_counts["TITLE"] + label_counts["UPPER"])
        ),
        "frequent_title_tokens": frequent_titles,
        "sample_TITLE": title_examples[: args.examples_per_label],
        "sample_UPPER": upper_examples[: args.examples_per_label],
        "notes": [
            "Review sample_TITLE/UPPER for wiki headers, link titles, and editorial caps.",
            "frequent_title_tokens with high count may indicate systematic noise.",
            "Sentence starts are IGNORE — model should learn names, not sentence case.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k not in ("sample_TITLE", "sample_UPPER")}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
