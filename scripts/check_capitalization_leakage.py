#!/usr/bin/env python3
"""Leakage / integrity checks for capitalization dataset + model input format."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_capitalization.casing import kurmanji_lower  # noqa: E402
from kurmanji_capitalization.constants import IGNORE_LABEL  # noqa: E402
from kurmanji_capitalization.dataset import (  # noqa: E402
    load_processed_jsonl,
    tokens_and_labels_from_article,
)
from kurmanji_capitalization.normalization import normalize_for_dataset  # noqa: E402
from kurmanji_capitalization.sentence_rule import (  # noqa: E402
    SentenceRuleConfig,
    sentence_start_word_indices,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed_capitalization"))
    p.add_argument("--raw", type=Path, default=Path("data/raw/wikipedia.jsonl"))
    p.add_argument("--max-check-articles", type=int, default=200)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/capitalization-xlm-r-base-v1/leakage_check.json"),
    )
    args = p.parse_args()

    splits = {
        name: load_processed_jsonl(args.processed_dir / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }
    id_sets = {name: {r["article_id"] for r in rows} for name, rows in splits.items()}

    overlaps = {}
    names = list(id_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = id_sets[names[i]] & id_sets[names[j]]
            overlaps[f"{names[i]}_x_{names[j]}"] = len(inter)

    # Near-duplicate article texts across splits (exact normalized text hash)
    from hashlib import sha1

    text_by_split: dict[str, dict[str, str]] = {n: {} for n in names}
    # sample articles from raw for hash of full article text
    want = set().union(*id_sets.values())
    with args.raw.open(encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            aid = obj["article_id"]
            if aid not in want:
                continue
            h = sha1(normalize_for_dataset(obj["text"]).encode("utf-8")).hexdigest()
            for n in names:
                if aid in id_sets[n]:
                    text_by_split[n][aid] = h

    hash_sets = {n: set(text_by_split[n].values()) for n in names}
    hash_overlaps = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            hash_overlaps[f"{names[i]}_x_{names[j]}"] = len(
                hash_sets[names[i]] & hash_sets[names[j]]
            )

    # Sentence-start IGNORE: within a window, after .?! the next letter-word must be IGNORE.
    # Do NOT treat window index 0 as a sentence start (windows may begin mid-sentence).
    from kurmanji_capitalization.casing import first_letter_index
    from kurmanji_capitalization.constants import SENTENCE_END_PUNCT

    n_after_punct_ok = 0
    n_after_punct_bad = 0
    n_checked = 0
    n_bad_case = 0
    label_counts = Counter()
    examples_bad_case = []

    for row in splits["train"][:5000] + splits["validation"][:1000] + splits["test"][:1000]:
        tokens = row["tokens"]
        labels = row["labels"]
        expect_start = False
        # If window starts mid-sentence we don't know; only enforce after in-window terminators.
        for i, (tok, lab) in enumerate(zip(tokens, labels)):
            label_counts[lab] += 1
            if tok in SENTENCE_END_PUNCT:
                expect_start = True
                continue
            if first_letter_index(tok) is None:
                continue
            if expect_start:
                if lab == IGNORE_LABEL:
                    n_after_punct_ok += 1
                else:
                    n_after_punct_bad += 1
                expect_start = False
            if lab == IGNORE_LABEL:
                continue
            n_checked += 1
            if tok != kurmanji_lower(tok):
                n_bad_case += 1
                if len(examples_bad_case) < 20:
                    examples_bad_case.append({"id": row["id"], "token": tok, "label": lab})

    # TITLE token overlap between train and test (surface form lowercased)
    def title_vocab(rows):
        v = Counter()
        for row in rows:
            for tok, lab in zip(row["tokens"], row["labels"]):
                if lab == "TITLE":
                    v[kurmanji_lower(tok)] += 1
        return v

    train_titles = title_vocab(splits["train"])
    test_titles = title_vocab(splits["test"])
    shared = set(train_titles) & set(test_titles)
    test_title_mass = sum(test_titles.values())
    shared_mass = sum(test_titles[t] for t in shared)

    report = {
        "article_id_overlaps": overlaps,
        "exact_text_hash_overlaps": hash_overlaps,
        "article_split_ok": all(v == 0 for v in overlaps.values()),
        "trainable_tokens_checked": n_checked,
        "trainable_not_lower": n_bad_case,
        "input_casing_ok": n_bad_case == 0,
        "sentence_start_after_punct_ok": n_after_punct_ok,
        "sentence_start_after_punct_bad": n_after_punct_bad,
        "sentence_starts_labeled_ignore": n_after_punct_bad == 0,
        "label_counts_sample": dict(label_counts),
        "title_vocab": {
            "train_unique": len(train_titles),
            "test_unique": len(test_titles),
            "shared_unique": len(shared),
            "test_title_tokens": test_title_mass,
            "shared_title_token_mass": shared_mass,
            "shared_mass_fraction": shared_mass / max(1, test_title_mass),
            "note": "High shared TITLE mass is expected for common geo names; use unseen-name probe.",
        },
        "examples_bad_case": examples_bad_case,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "article_split_ok": report["article_split_ok"],
        "input_casing_ok": report["input_casing_ok"],
        "sentence_starts_labeled_ignore": report["sentence_starts_labeled_ignore"],
        "article_id_overlaps": report["article_id_overlaps"],
        "exact_text_hash_overlaps": report["exact_text_hash_overlaps"],
        "title_vocab": report["title_vocab"],
        "trainable_not_lower": report["trainable_not_lower"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
