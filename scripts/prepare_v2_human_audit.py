#!/usr/bin/env python3
"""Prepare a blind human-audit CSV from the stratified manual_review sample.

Does NOT include assisted-review decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2_question_processed"

FIELDS = [
    "record_id",
    "source",
    "kurmanji_ok",
    "question_is_real",
    "punctuation_ok",
    "context_ok",
    "encoding_ok",
    "duplicate_or_template",
    "accept",
    "comment",
    "reviewer",
    "reviewed_at",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sample",
        type=Path,
        default=OUT / "manual_review_sample.jsonl",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=OUT / "human_audit_template.csv",
    )
    args = p.parse_args()

    rows = [
        json.loads(line)
        for line in args.sample.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "record_id": r["record_id"],
                    "source": r["source"],
                    "kurmanji_ok": "",
                    "question_is_real": "",
                    "punctuation_ok": "",
                    "context_ok": "",
                    "encoding_ok": "",
                    "duplicate_or_template": "",
                    "accept": "",
                    "comment": "",
                    "reviewer": "",
                    "reviewed_at": "",
                }
            )
    # Companion text snippets for the reviewer (separate file; still no assisted labels)
    preview = OUT / "human_audit_texts.jsonl"
    with preview.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "record_id": r["record_id"],
                        "source": r["source"],
                        "word_count": r.get("word_count"),
                        "question_count": r.get("question_count"),
                        "text": r.get("text"),
                        "license": r.get("license"),
                        "source_url": r.get("source_url"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {args.output} n={len(rows)}")
    print(f"wrote {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
