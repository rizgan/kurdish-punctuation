#!/usr/bin/env python3
"""Check v2 question train corpus for leakage into frozen validation/test (Stage 4).

Compares against:
  - data/processed/validation.jsonl
  - data/processed/test.jsonl
  - long-text 400 test articles (same test.jsonl article texts)
  - specialized question-test (if present)

Checks: exact hash, nopunct hash, lowercased hash, n-gram Jaccard, long word sequences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_V1 = ROOT / "data" / "processed"
V2_Q = ROOT / "data" / "v2_question_processed"
SPECIALIZED = ROOT / "data" / "test_question_specialized"

WORD_RE = re.compile(r"[A-Za-zÀ-öø-ÿĀ-ſƀ-ɏçêîşûÇÊÎŞÛğĞıİöÖüÜ']+", re.U)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def normalize_exact(text: str) -> str:
    return collapse_ws(nfc(text))


def normalize_lower(text: str) -> str:
    return normalize_exact(text).lower()


def normalize_nopunct(text: str) -> str:
    t = normalize_lower(text)
    t = re.sub(r"[,.?!;:…]+", "", t)
    return collapse_ws(t)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def char_ngrams(text: str, n: int = 5) -> set[str]:
    t = normalize_nopunct(text)
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def load_jsonl_texts(path: Path, text_keys: tuple[str, ...] = ("text", "tokens")) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text")
            if text is None and "tokens" in obj:
                toks = obj["tokens"]
                if isinstance(toks, list):
                    text = " ".join(str(t) for t in toks)
            if not text:
                continue
            rows.append(
                {
                    "ref_id": obj.get("article_id") or obj.get("record_id") or obj.get("id") or f"{path.name}:{i}",
                    "split_file": str(path),
                    "text": str(text),
                }
            )
    return rows


def load_holdout_refs() -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    # Frozen v1 holdouts only (do not include train or unrelated large dumps like dev.jsonl)
    for name in ("validation.jsonl", "test.jsonl"):
        refs.extend(load_jsonl_texts(PROCESSED_V1 / name))
    # Specialized question-test (documents or contexts)
    if SPECIALIZED.exists():
        for path in SPECIALIZED.rglob("*.jsonl"):
            refs.extend(load_jsonl_texts(path))
    return refs


def minhash_signature(text: str, *, ngram_n: int = 5, num_hashes: int = 64) -> tuple[int, ...]:
    t = normalize_nopunct(text)
    grams = {t[i : i + ngram_n] for i in range(max(0, len(t) - ngram_n + 1))} if t else set()
    if not grams:
        return tuple(0 for _ in range(num_hashes))
    vals = [int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16) for g in grams]
    sig: list[int] = []
    for seed in range(num_hashes):
        a = 2 * seed + 1
        b = seed * 0x9E3779B9
        best = min(((a * x + b) & 0xFFFFFFFF) for x in vals)
        sig.append(best)
    return tuple(sig)


def minhash_jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def build_ref_indexes(refs: list[dict[str, Any]]) -> dict[str, Any]:
    exact: dict[str, str] = {}
    lower: dict[str, str] = {}
    nopunct: dict[str, str] = {}
    seq_index: dict[str, set[str]] = defaultdict(set)
    gram_store: list[tuple[str, tuple[int, ...]]] = []

    for r in refs:
        t = r["text"]
        rid = str(r["ref_id"])
        exact[sha(normalize_exact(t))] = rid
        lower[sha(normalize_lower(t))] = rid
        nopunct[sha(normalize_nopunct(t))] = rid
        gram_store.append((rid, minhash_signature(t)))
        w = words(t)
        for n in (12, 16, 20):
            if len(w) < n:
                continue
            for i in range(0, len(w) - n + 1, max(1, n // 2)):
                seq = " ".join(w[i : i + n])
                seq_index[seq].add(rid)

    return {
        "exact": exact,
        "lower": lower,
        "nopunct": nopunct,
        "gram_store": gram_store,
        "seq_index": seq_index,
    }


def check_train(
    train_rows: list[dict[str, Any]],
    indexes: dict[str, Any],
    *,
    near_threshold: float = 0.90,
) -> dict[str, Any]:
    exact_leaks = 0
    lower_leaks = 0
    nopunct_leaks = 0
    near_leaks = 0
    seq_leaks = 0
    removed_ids: list[str] = []
    leak_details: list[dict[str, Any]] = []

    clean: list[dict[str, Any]] = []
    for row in train_rows:
        text = row.get("text") or ""
        rid = row.get("record_id")
        reasons: list[str] = []
        matched_ref = None

        eh = sha(normalize_exact(text))
        if eh in indexes["exact"]:
            exact_leaks += 1
            reasons.append("exact")
            matched_ref = indexes["exact"][eh]

        lh = sha(normalize_lower(text))
        if lh in indexes["lower"]:
            lower_leaks += 1
            reasons.append("lower")
            matched_ref = matched_ref or indexes["lower"][lh]

        nh = sha(normalize_nopunct(text))
        if nh in indexes["nopunct"]:
            nopunct_leaks += 1
            reasons.append("nopunct")
            matched_ref = matched_ref or indexes["nopunct"][nh]

        # long word sequence overlap
        w = words(text)
        seq_hit = False
        for n in (12, 16, 20):
            if len(w) < n:
                continue
            for i in range(0, len(w) - n + 1, max(1, n // 2)):
                seq = " ".join(w[i : i + n])
                refs_hit = indexes["seq_index"].get(seq)
                if refs_hit:
                    seq_hit = True
                    matched_ref = matched_ref or next(iter(refs_hit))
                    break
            if seq_hit:
                break
        if seq_hit:
            seq_leaks += 1
            reasons.append("long_word_sequence")

        # near duplicate vs holdout via MinHash
        grams = minhash_signature(text)
        near_hit = False
        if grams and not reasons:
            for ref_id, ref_grams in indexes["gram_store"]:
                if minhash_jaccard(grams, ref_grams) >= near_threshold:
                    near_hit = True
                    matched_ref = matched_ref or ref_id
                    break
        if near_hit:
            near_leaks += 1
            reasons.append("near")

        if reasons:
            removed_ids.append(str(rid))
            leak_details.append(
                {
                    "record_id": rid,
                    "reasons": reasons,
                    "matched_ref": matched_ref,
                }
            )
        else:
            clean.append(row)

    contamination = bool(removed_ids)
    # test_contamination true if any leak into holdout (plan: forbid experiment)
    report = {
        "checked_train_records": len(train_rows),
        "exact_leaks": exact_leaks,
        "normalized_leaks": lower_leaks + nopunct_leaks,
        "lower_leaks": lower_leaks,
        "nopunct_leaks": nopunct_leaks,
        "near_leaks_removed": near_leaks,
        "long_sequence_leaks": seq_leaks,
        "removed_record_ids": removed_ids,
        "leak_details": leak_details[:200],
        "clean_train_records": len(clean),
        "test_contamination": contamination,
    }
    return report, clean


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train",
        type=Path,
        default=V2_Q / "train_questions.jsonl",
    )
    p.add_argument(
        "--accepted",
        type=Path,
        default=V2_Q / "accepted.jsonl",
    )
    p.add_argument("--near-threshold", type=float, default=0.90)
    p.add_argument(
        "--write-clean",
        action="store_true",
        help="Rewrite accepted/train JSONL without leaked records",
    )
    args = p.parse_args()

    refs = load_holdout_refs()
    print(f"[info] holdout reference texts: {len(refs)}")
    indexes = build_ref_indexes(refs)

    train_path = args.train if args.train.exists() else args.accepted
    train_rows: list[dict[str, Any]] = []
    with train_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                train_rows.append(json.loads(line))

    report, clean = check_train(train_rows, indexes, near_threshold=args.near_threshold)
    out = V2_Q / "leakage_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "leak_details"}, ensure_ascii=False, indent=2))
    print(f"[wrote] {out}")

    if args.write_clean and report["removed_record_ids"]:
        removed = set(report["removed_record_ids"])
        for path in (args.accepted, args.train):
            if not path.exists():
                continue
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            kept = [r for r in rows if str(r.get("record_id")) not in removed]
            path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                encoding="utf-8",
            )
            print(f"[rewrote] {path} kept={len(kept)} removed={len(rows) - len(kept)}")

    if report["test_contamination"]:
        print("[gate] FAIL: test_contamination=true — remove leaks before any experiment")
        return 2
    print("[gate] PASS: test_contamination=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
