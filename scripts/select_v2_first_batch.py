#!/usr/bin/env python3
"""Select first-batch train questions (2k–4k) with diversity constraints.

Plan Stage 13: add 2000–4000 real unique questions first — do not dump tens of thousands.
Also reclassify question types with corrected markers and refresh statistics.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2_question_processed"

# Import corrected classifier from filter module
import sys

sys.path.insert(0, str(ROOT / "scripts"))
from filter_v2_question_corpus import classify_question_type  # noqa: E402


def load_accepted(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score_record(r: dict) -> float:
    """Higher = better candidate for first batch."""
    wc = int(r.get("word_count") or 0)
    qc = int(r.get("question_count") or 0)
    text = r.get("text") or ""
    score = 0.0
    # Prefer multi-sentence contexts in the 20–250 word band
    if 20 <= wc <= 250:
        score += 5.0
    elif 10 <= wc < 20:
        score += 1.0
    elif wc > 250:
        score += 2.0
    # Prefer 1–3 questions (not huge FAQ dumps)
    if 1 <= qc <= 3:
        score += 3.0
    elif qc > 3:
        score += 0.5
    # Prefer natural context (statement + question)
    if re.search(r"[.!] .+\?", text):
        score += 2.0
    if r.get("needs_context"):
        score -= 3.0
    if r.get("turkish_review_flag"):
        score -= 1.0
    if r.get("context_constructed"):
        score -= 2.0
    # Prefer sources with clearer licenses slightly
    if r.get("source") == "kurdish_ai_corpus":
        score += 0.5
    if r.get("source") == "tatoeba":
        score -= 1.5  # short; needs context construction
    return score


def select_batch(
    rows: list[dict],
    *,
    target: int = 3000,
    seed: int = 42,
    max_type_share: float = 0.40,
    source_caps: dict[str, int] | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    source_caps = source_caps or {
        "kurmanji_news": 1800,
        "kurdish_ai_corpus": 1500,
        "tatoeba": 200,
    }

    # Reclassify
    for r in rows:
        r["question_type_auto"] = classify_question_type(r.get("text") or "")
        r["_score"] = score_record(r)

    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    for src in by_src:
        by_src[src].sort(key=lambda x: x["_score"], reverse=True)

    selected: list[dict] = []
    type_counts: Counter = Counter()
    src_counts: Counter = Counter()

    # Round-robin across sources by ranked lists
    pointers = {s: 0 for s in by_src}
    sources = [s for s in ("kurmanji_news", "kurdish_ai_corpus", "tatoeba") if s in by_src]
    while len(selected) < target:
        progressed = False
        for src in sources:
            if len(selected) >= target:
                break
            if src_counts[src] >= source_caps.get(src, target):
                continue
            pool = by_src[src]
            i = pointers[src]
            while i < len(pool):
                cand = pool[i]
                i += 1
                qtype = cand["question_type_auto"]
                # Enforce type diversity once we have enough samples
                if len(selected) >= 200:
                    if (type_counts[qtype] + 1) / (len(selected) + 1) > max_type_share:
                        # try to skip over-represented type
                        continue
                selected.append(cand)
                type_counts[qtype] += 1
                src_counts[src] += 1
                progressed = True
                break
            pointers[src] = i
        if not progressed:
            break

    rng.shuffle(selected)
    for r in selected:
        r.pop("_score", None)
        r["split"] = "train"
        r["first_batch"] = True
    return selected


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    accepted_path = OUT / "accepted.jsonl"
    rows = load_accepted(accepted_path)
    print(f"accepted={len(rows)}")

    # Refresh type stats on full accepted
    full_types = Counter(classify_question_type(r.get("text") or "") for r in rows)
    for r in rows:
        r["question_type_auto"] = classify_question_type(r.get("text") or "")

    # Rewrite accepted with corrected types
    with accepted_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    batch = select_batch(rows, target=args.target, seed=args.seed)
    train_path = OUT / "train_questions.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for r in batch:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    batch_types = Counter(r["question_type_auto"] for r in batch)
    batch_src = Counter(r["source"] for r in batch)
    needs_ctx = sum(1 for r in batch if r.get("needs_context"))
    constructed_share = needs_ctx / len(batch) if batch else 0.0

    stats = {
        "full_accepted": len(rows),
        "full_question_types": dict(full_types),
        "full_max_type_share": (max(full_types.values()) / len(rows) if rows else 0),
        "first_batch_size": len(batch),
        "first_batch_by_source": dict(batch_src),
        "first_batch_question_types": dict(batch_types),
        "first_batch_max_type_share": (
            max(batch_types.values()) / len(batch) if batch else 0
        ),
        "first_batch_needs_context_share": constructed_share,
        "gate": {
            "batch_in_2000_4000": 2000 <= len(batch) <= 4000,
            "type_share_le_40": (
                (max(batch_types.values()) / len(batch) <= 0.40) if batch else False
            ),
            "needs_context_share_note": (
                "Tatoeba short items marked needs_context; "
                "constructed contexts must stay <=20% after context building"
            ),
        },
    }
    (OUT / "question_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
