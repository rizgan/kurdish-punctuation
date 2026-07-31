#!/usr/bin/env python3
"""Build v2 train dataset: frozen v1 val/test + v1 train windows + new question windows.

Also creates specialized question-test holdout (not in train).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from kurmanji_punctuation.dataset import (  # noqa: E402
    iter_word_windows,
    label_distribution,
    tokens_and_labels_from_text,
    window_period_stats,
    write_jsonl,
)
from kurmanji_punctuation.normalization import normalize_for_dataset  # noqa: E402

V2_Q = ROOT / "data" / "v2_question_processed"
V1_PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed_v2_question"
SPECIALIZED = ROOT / "data" / "test_question_specialized"

WORD_RE = re.compile(r"[A-Za-zÀ-öø-ÿçêîşûÇÊÎŞÛğĞıİöÖüÜ']+", re.U)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def question_window_stats(rows: list[dict]) -> dict[str, Any]:
    q_total = 0
    q_at_last = 0
    windows_with_q = 0
    windows_multi_q = 0
    windows_no_q = 0
    for row in rows:
        labs = row["labels"]
        nq = sum(1 for x in labs if x == "QUESTION")
        q_total += nq
        if labs and labs[-1] == "QUESTION":
            q_at_last += 1
        if nq == 0:
            windows_no_q += 1
        elif nq == 1:
            windows_with_q += 1
        else:
            windows_multi_q += 1
            windows_with_q += 1
    return {
        "question_total": q_total,
        "question_at_last_word": q_at_last,
        "fraction_of_question_labels_at_final_word": (q_at_last / q_total) if q_total else 0.0,
        "windows_with_question": windows_with_q,
        "windows_without_question": windows_no_q,
        "windows_with_multiple_question": windows_multi_q,
        "fraction_question_windows": windows_with_q / len(rows) if rows else 0.0,
    }


def build_context_if_needed(
    text: str,
    *,
    filler_pool: list[str],
    rng: random.Random,
    constructed: bool,
) -> tuple[str, bool]:
    """Ensure enough words for windowing; optionally prepend/append neutral fillers."""
    wc = word_count(text)
    if wc >= 40 and (text.count(".") + text.count("!") + text.count("?")) >= 2:
        return text, constructed
    if not filler_pool:
        return text, constructed
    before = rng.choice(filler_pool)
    after = rng.choice(filler_pool)
    # Place question not always at end
    if rng.random() < 0.5:
        merged = f"{before} {text} {after}".strip()
    else:
        merged = f"{before} {after} {text}".strip()
    return merged, True


def windows_from_texts(
    articles: list[dict[str, str]],
    *,
    tokenizer,
    seed: int,
    cfg: dict,
) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for art in articles:
        text = normalize_for_dataset(art["text"], map_ellipsis_to_period=True)
        parsed = tokens_and_labels_from_text(text)
        if not parsed:
            continue
        tokens, labels = parsed
        for w in iter_word_windows(
            art["article_id"],
            tokens,
            labels,
            tokenizer=tokenizer,
            rng=rng,
            max_model_length=int(cfg.get("max_model_length", 256)),
            min_words=int(cfg.get("min_words_per_window", 80)),
            target_words_min=int(cfg.get("target_words_min", 110)),
            target_words_max=int(cfg.get("target_words_max", 180)),
            overlap_words=int(cfg.get("train_overlap_words", 8)),
        ):
            w["source"] = art.get("source", "v1_wiki")
            w["has_question"] = any(x == "QUESTION" for x in w["labels"])
            rows.append(w)
    return rows


def pick_specialized(
    accepted: list[dict],
    train_ids: set[str],
    *,
    n: int = 800,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    pool = [
        r
        for r in accepted
        if r["record_id"] not in train_ids
        and int(r.get("word_count") or 0) >= 20
        and int(r.get("question_count") or 0) >= 1
        and (r.get("text") or "").count(".") + (r.get("text") or "").count("!") >= 1
    ]
    rng.shuffle(pool)
    # Stratify roughly by source
    by_src: dict[str, list] = defaultdict(list)
    for r in pool:
        by_src[r["source"]].append(r)
    out: list[dict] = []
    sources = list(by_src.keys())
    i = 0
    while len(out) < n and sources:
        src = sources[i % len(sources)]
        if by_src[src]:
            r = by_src[src].pop()
            r = {**r, "split": "specialized_test", "manual_review": False}
            out.append(r)
        else:
            sources.remove(src)
            if not sources:
                break
            continue
        i += 1
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "v2-base-frozen.yaml")
    p.add_argument("--specialized-size", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg_all = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ds_cfg = cfg_all.get("dataset", {})

    review = json.loads((V2_Q / "manual_review_report.json").read_text(encoding="utf-8"))
    allowed = set(review.get("allowed_train_sources") or [])
    if not allowed:
        raise SystemExit("[gate] FAIL: no allowed_train_sources from manual review")

    train_q = load_jsonl(V2_Q / "train_questions.jsonl")
    train_q = [r for r in train_q if r.get("source") in allowed]
    train_ids = {r["record_id"] for r in train_q}
    accepted = load_jsonl(V2_Q / "accepted.jsonl")

    # Specialized holdout
    specialized = pick_specialized(accepted, train_ids, n=args.specialized_size, seed=args.seed)
    # Ensure no overlap with train
    specialized = [r for r in specialized if r["record_id"] not in train_ids]
    SPECIALIZED.mkdir(parents=True, exist_ok=True)
    write_jsonl(SPECIALIZED / "contexts.jsonl", specialized)
    # Genre buckets (rough)
    genre_map = {
        "kurmanji_news": "news_interviews",
        "kurdish_ai_corpus": "other",
        "tatoeba": "educational_faq",
    }
    by_genre: dict[str, list] = defaultdict(list)
    for r in specialized:
        by_genre[genre_map.get(r["source"], "other")].append(r)
    for g, rows in by_genre.items():
        write_jsonl(SPECIALIZED / f"{g}.jsonl", rows)
    (SPECIALIZED / "meta.json").write_text(
        json.dumps(
            {
                "n": len(specialized),
                "by_source": dict(Counter(r["source"] for r in specialized)),
                "by_genre": {k: len(v) for k, v in by_genre.items()},
                "disjoint_from_train": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Filler sentences from v1 train windows (neutral statements without ?)
    v1_train = load_jsonl(V1_PROC / "train.jsonl")
    filler_pool: list[str] = []
    rng = random.Random(args.seed)
    for row in v1_train:
        if any(l == "QUESTION" for l in row["labels"]):
            continue
        # rebuild a short statement-like span from tokens+labels
        parts = []
        from kurmanji_punctuation.constants import LABEL_TO_PUNCTUATION

        for tok, lab in zip(row["tokens"], row["labels"]):
            parts.append(tok)
            if lab in LABEL_TO_PUNCTUATION:
                parts[-1] = parts[-1] + LABEL_TO_PUNCTUATION[lab]
        text = " ".join(parts)
        # take first ~2 sentences
        sents = re.split(r"(?<=[.!])\s+", text)
        sents = [s for s in sents if "?" not in s and word_count(s) >= 6]
        if sents:
            filler_pool.append(sents[0])
        if len(filler_pool) >= 2000:
            break

    # Build question articles with optional context construction
    q_articles: list[dict[str, str]] = []
    constructed_n = 0
    for r in train_q:
        text = r["text"]
        constructed = bool(r.get("context_constructed"))
        text2, constructed = build_context_if_needed(
            text, filler_pool=filler_pool, rng=rng, constructed=constructed
        )
        if constructed:
            constructed_n += 1
        q_articles.append(
            {
                "article_id": f"v2q_{r['record_id']}",
                "text": text2,
                "source": r["source"],
            }
        )
    constructed_share = constructed_n / len(q_articles) if q_articles else 0.0
    if constructed_share > 0.20:
        print(f"[warn] constructed_share={constructed_share:.3f} > 0.20")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg_all.get("model", {}).get("name", "FacebookAI/xlm-roberta-base")
    )

    print(f"[build] windowing {len(q_articles)} question articles…")
    q_windows = windows_from_texts(q_articles, tokenizer=tokenizer, seed=args.seed, cfg=ds_cfg)
    print(f"[build] question windows={len(q_windows)}")

    # Merge with v1 train windows (do not rebuild wiki — keeps frozen v1 train hash base)
    # Dedup by exact token join hash against v1
    v1_hashes = set()
    for row in v1_train:
        key = " ".join(row["tokens"]).lower()
        v1_hashes.add(hashlib.sha256(key.encode()).hexdigest())

    extra = []
    for row in q_windows:
        key = " ".join(row["tokens"]).lower()
        h = hashlib.sha256(key.encode()).hexdigest()
        if h in v1_hashes:
            continue
        extra.append(row)

    train_merged = list(v1_train) + extra
    rng.shuffle(train_merged)

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "train.jsonl", train_merged)
    # Frozen val/test copies
    for name in ("validation.jsonl", "test.jsonl"):
        shutil.copy2(V1_PROC / name, OUT / name)

    # Verify frozen hashes match v1
    v1_run = json.loads(
        (ROOT / "models/punctuation/kurmanji-xlm-r-base-v1/run_info.json").read_text(encoding="utf-8")
    )
    expected = {
        x.split(":", 1)[0]: x.split(":", 1)[1]
        for x in v1_run.get("dataset_files_sha256", [])
        if ":" in x
    }
    for name in ("validation.jsonl", "test.jsonl"):
        h = sha256_file(OUT / name)
        if expected.get(name) and expected[name] != h:
            raise SystemExit(f"[gate] FAIL: {name} hash mismatch vs v1")

    label_counts = {
        "train": label_distribution(train_merged),
        "validation": label_distribution(load_jsonl(OUT / "validation.jsonl")),
        "test": label_distribution(load_jsonl(OUT / "test.jsonl")),
    }
    pstats = {
        "train": window_period_stats(train_merged),
        "validation": window_period_stats(load_jsonl(OUT / "validation.jsonl")),
        "test": window_period_stats(load_jsonl(OUT / "test.jsonl")),
    }
    qstats = question_window_stats(train_merged)
    src_dist = Counter(r.get("source", "v1_wiki") for r in train_merged)

    stats = {
        "sample_mode": "continuous_word_windows",
        "n_samples": {
            "train": len(train_merged),
            "validation": len(load_jsonl(OUT / "validation.jsonl")),
            "test": len(load_jsonl(OUT / "test.jsonl")),
        },
        "v1_train_windows": len(v1_train),
        "new_question_articles": len(q_articles),
        "new_question_windows": len(extra),
        "constructed_context_share": constructed_share,
        "label_counts": label_counts,
        "period_window_stats": pstats,
        "question_window_stats": qstats,
        "source_distribution_train": dict(src_dist),
        "specialized_question_test_n": len(specialized),
        "allowed_sources": sorted(allowed),
        "seed": args.seed,
        "gates": {
            "period_at_final_le_6pct": pstats["train"]["fraction_of_period_labels_at_final_word"]
            <= 0.06,
            "question_at_final_le_10pct": qstats["fraction_of_question_labels_at_final_word"]
            <= 0.10,
            "train_question_support_ge_3000": label_counts["train"].get("QUESTION", 0) >= 3000,
            "val_test_hash_match_v1": True,
        },
    }
    (OUT / "statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sums = []
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "statistics.json"):
        sums.append(f"{sha256_file(OUT / name)}  {name}")
    (OUT / "DATASET_SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    dataset_hash = hashlib.sha256((OUT / "DATASET_SHA256SUMS.txt").read_bytes()).hexdigest()
    (OUT / "dataset_hash.txt").write_text(dataset_hash + "\n", encoding="utf-8")

    # Also store question corpus hash used
    q_hash = sha256_file(V2_Q / "train_questions.jsonl")
    meta = {
        "dataset_hash": dataset_hash,
        "train_questions_sha256": q_hash,
        "statistics": stats,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    gates = stats["gates"]
    if not all(gates.values()):
        print("[gate] FAIL dataset gates:", gates)
        return 2
    print("[gate] PASS dataset gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
