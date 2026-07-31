#!/usr/bin/env python3
"""Filter raw v2 question corpora into accepted/rejected JSONL (Stage 2).

Checks run in plan order:
  1. script / Arabic ratio
  2. characteristic letter stats (informational)
  3. Turkish-letter flag for manual review
  4. real question-mark validation
  5. length (min 5 words; prefer 20–250)
  6. exact / punctuation-stripped / near dedup
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "v2_question_raw"
OUT = ROOT / "data" / "v2_question_processed"

ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
LATIN_LETTER_RE = re.compile(r"[A-Za-zÀ-öø-ÿĀ-ſƀ-ɏ]")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
WIKI_TEMPLATE_RE = re.compile(r"\{\{[^}]+\}\}|\[\[[^\]]+\]\]")
WORD_RE = re.compile(r"[A-Za-zÀ-öø-ÿĀ-ſƀ-ɏçêîşûÇÊÎŞÛğĞıİöÖüÜ']+", re.U)

KMR_CHARS = set("çêîşûÇÊÎŞÛ")
TR_CHARS = set("ğıİöüĞ")

QUESTION_MARKERS = [
    ("WHO", re.compile(r"\bkî\b", re.I)),
    ("WHAT", re.compile(r"\bçi\b", re.I)),
    # Do NOT use bare \bku\b — it matches the relative/complementizer "ku" everywhere.
    ("WHERE", re.compile(r"\bli\s+ku\b|\bkuder[eê]?\b|\bli\s+kû\b", re.I)),
    ("WHEN", re.compile(r"\bkengî\b|\bkengê\b", re.I)),
    ("WHY", re.compile(r"\bçima\b", re.I)),
    ("HOW", re.compile(r"\bçawa\b", re.I)),
    ("HOW_MANY", re.compile(r"\bçend\b", re.I)),
    ("MA", re.compile(r"(?:^|[.!?]\s+)\s*ma\b|\bma\s+\S+\s+\S+\?", re.I)),
    ("GELO", re.compile(r"\bgelo\b", re.I)),
]


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def strip_outer_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in "\"'“”«»" and s[-1] in "\"'“”«»":
        return s[1:-1].strip()
    return s


def normalize_exact(text: str) -> str:
    return collapse_ws(strip_outer_quotes(nfc(text)))


def normalize_nopunct(text: str) -> str:
    t = normalize_exact(text).lower()
    t = re.sub(r"[,.?!;:…]+", "", t)
    return collapse_ws(t)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def script_stats(text: str) -> dict[str, float]:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return {"arabic_ratio": 1.0, "latin_ratio": 0.0, "n_letters": 0}
    n = len(letters)
    arabic = sum(1 for ch in letters if ARABIC_RE.match(ch))
    latin = sum(1 for ch in letters if LATIN_LETTER_RE.match(ch))
    return {
        "arabic_ratio": arabic / n,
        "latin_ratio": latin / n,
        "n_letters": n,
    }


def characteristic_counts(text: str) -> dict[str, int]:
    return {
        "kmr_special_chars": sum(1 for ch in text if ch in KMR_CHARS),
        "turkish_special_chars": sum(1 for ch in text if ch in TR_CHARS),
    }


def question_mark_ok(text: str) -> tuple[bool, str]:
    if "?" not in text:
        return False, "no_question_mark"
    # Strip URLs then re-check
    stripped = URL_RE.sub(" ", text)
    if "?" not in stripped:
        return False, "question_mark_only_in_url"
    if re.fullmatch(r"[?\s]+", stripped.strip()):
        return False, "only_question_marks"
    if "???" in text:
        return False, "triple_question_marks"
    if WIKI_TEMPLATE_RE.search(text) and text.count("?") == text.count("?"):  # keep soft
        # Reject obvious template-only stubs
        if len(WORD_RE.findall(text)) < 5 and "{{" in text:
            return False, "wiki_template"
    return True, "ok"


def classify_question_type(text: str) -> str:
    hits: list[str] = []
    for name, rx in QUESTION_MARKERS:
        if rx.search(text):
            hits.append(name)
    uniq: list[str] = []
    for h in hits:
        if h not in uniq:
            uniq.append(h)
    if len(uniq) > 1:
        return "MULTIPLE_QUESTION"
    if len(uniq) == 1:
        return uniq[0]
    if "?" in text:
        return "YES_NO"
    return "OTHER"


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


def minhash_signature(text: str, *, ngram_n: int = 5, num_hashes: int = 64) -> tuple[int, ...]:
    """Compact MinHash over character n-grams for near-duplicate detection."""
    grams = char_ngrams(text, ngram_n)
    if not grams:
        return tuple(0 for _ in range(num_hashes))
    # Stable 32-bit hashes of grams
    vals = [int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16) for g in grams]
    sig: list[int] = []
    for seed in range(num_hashes):
        # Simple independent hash family: (a*x + b) mod 2^32
        a = 2 * seed + 1
        b = seed * 0x9E3779B9
        best = min(((a * x + b) & 0xFFFFFFFF) for x in vals)
        sig.append(best)
    return tuple(sig)


def minhash_jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)

@dataclass
class FilterStats:
    raw_records: int = 0
    accepted: int = 0
    rejected: int = 0
    reject_reasons: Counter = field(default_factory=Counter)
    exact_duplicates_removed: int = 0
    punctuation_only_duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    by_source_raw: Counter = field(default_factory=Counter)
    by_source_accepted: Counter = field(default_factory=Counter)
    question_types: Counter = field(default_factory=Counter)
    kmr_char_docs: int = 0
    turkish_flagged: int = 0


def load_raw_jsonls(sources: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mapping = {
        "tatoeba": RAW / "tatoeba" / "questions_raw.jsonl",
        "kurmanji_news": RAW / "kurmanji_news" / "questions_raw.jsonl",
        "kurdish_ai_corpus": RAW / "kurdish_ai_corpus" / "questions_raw.jsonl",
        "opensubtitles": RAW / "opensubtitles" / "questions_raw.jsonl",
        "kurcorpus": RAW / "kurcorpus" / "questions_raw.jsonl",
    }
    for src in sources:
        path = mapping.get(src)
        if path is None or not path.exists():
            print(f"[warn] missing raw file for {src}: {path}")
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj.setdefault("source", src)
                rows.append(obj)
        print(f"[load] {src}: {path} -> loaded batch")
    return rows


def basic_reject(rec: dict[str, Any], stats: FilterStats) -> str | None:
    text = rec.get("text") or ""
    if not text.strip():
        return "empty"
    st = script_stats(text)
    if st["n_letters"] == 0:
        return "no_letters"
    if st["arabic_ratio"] > 0.01:
        return "arabic_script"
    if st["latin_ratio"] < 0.80:
        return "low_latin_ratio"
    q_ok, reason = question_mark_ok(text)
    if not q_ok:
        return reason
    wc = word_count(text)
    if wc < 5:
        return "too_short"
    # Prefer longer contexts later; keep short Tatoeba for review pool but mark
    return None


def extract_question_contexts(text: str, max_chars: int = 1200) -> str:
    """Keep surrounding sentences around first '?'; prefer multi-sentence context."""
    text = collapse_ws(nfc(text))
    if len(text) <= max_chars:
        return text
    idx = text.find("?")
    if idx < 0:
        return text[:max_chars]
    start = max(0, idx - max_chars // 2)
    end = min(len(text), start + max_chars)
    return text[start:end].strip()


def filter_corpus(
    records: list[dict[str, Any]],
    *,
    near_dup_threshold: float = 0.90,
    near_dup_bucket_size: int = 2000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], FilterStats]:
    stats = FilterStats()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    seen_exact: set[str] = set()
    seen_nopunct: set[str] = set()
    # MinHash near-dup within source + length buckets (fast approximate Jaccard ≥ threshold)
    near_index: dict[tuple[str, int], list[tuple[int, ...]]] = defaultdict(list)

    for rec_i, rec in enumerate(records):
        if rec_i and rec_i % 2000 == 0:
            print(
                f"[filter] progress {rec_i}/{len(records)} "
                f"accepted={stats.accepted} rejected={stats.rejected}",
                flush=True,
            )
        stats.raw_records += 1
        src = rec.get("source", "unknown")
        stats.by_source_raw[src] += 1
        text = extract_question_contexts(str(rec.get("text") or ""))
        rec = {**rec, "text": text}

        reason = basic_reject(rec, stats)
        if reason:
            stats.rejected += 1
            stats.reject_reasons[reason] += 1
            rejected.append({**rec, "reject_reason": reason})
            continue

        chars = characteristic_counts(text)
        if chars["kmr_special_chars"] > 0:
            stats.kmr_char_docs += 1
        turkish_flag = chars["turkish_special_chars"] >= 3
        if turkish_flag:
            stats.turkish_flagged += 1

        exact_key = sha256_text(normalize_exact(text))
        if exact_key in seen_exact:
            stats.exact_duplicates_removed += 1
            stats.rejected += 1
            stats.reject_reasons["exact_duplicate"] += 1
            rejected.append({**rec, "reject_reason": "exact_duplicate"})
            continue

        nopunct_key = sha256_text(normalize_nopunct(text))
        if nopunct_key in seen_nopunct:
            stats.punctuation_only_duplicates_removed += 1
            stats.rejected += 1
            stats.reject_reasons["punctuation_only_duplicate"] += 1
            rejected.append({**rec, "reject_reason": "punctuation_only_duplicate"})
            continue

        sig = minhash_signature(text)
        wc = word_count(text)
        bucket = (src, wc // 20)
        near_hit = False
        for prev_sig in near_index[bucket][-near_dup_bucket_size:]:
            if minhash_jaccard(sig, prev_sig) >= near_dup_threshold:
                near_hit = True
                break
        if near_hit:
            stats.near_duplicates_removed += 1
            stats.rejected += 1
            stats.reject_reasons["near_duplicate"] += 1
            rejected.append({**rec, "reject_reason": "near_duplicate"})
            continue

        seen_exact.add(exact_key)
        seen_nopunct.add(nopunct_key)
        near_index[bucket].append(sig)

        qtype = classify_question_type(text)
        stats.question_types[qtype] += 1
        qcount = text.count("?")
        out = {
            "record_id": f"v2q_{len(accepted) + 1:06d}",
            "source": src,
            "source_document_id": rec.get("source_document_id"),
            "source_url": rec.get("source_url"),
            "text": text,
            "question_count": qcount,
            "word_count": wc,
            "question_type_auto": qtype,
            "verified_language": False,
            "verified_punctuation": False,
            "manual_review": False,
            "turkish_review_flag": turkish_flag,
            "kmr_special_char_count": chars["kmr_special_chars"],
            "license": rec.get("license"),
            "retrieved_at": rec.get("retrieved_at") or date.today().isoformat(),
            "context_constructed": bool(rec.get("context_constructed", False)),
            "split": "train",
            "raw_record_id": rec.get("record_id"),
        }
        # Prefer multi-sentence; mark short single-sentence for later context building
        if text.count(".") + text.count("!") + text.count("?") < 2:
            out["needs_context"] = True
        accepted.append(out)
        stats.accepted += 1
        stats.by_source_accepted[src] += 1

    return accepted, rejected, stats


def stratified_manual_sample(
    accepted: list[dict[str, Any]],
    *,
    total: int = 400,
    min_per_source: int = 50,
    seed: int = 42,
) -> list[dict[str, Any]]:
    import random

    rng = random.Random(seed)
    by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in accepted:
        by_src[r["source"]].append(r)

    sample: list[dict[str, Any]] = []
    sources = sorted(by_src.keys())
    if not sources:
        return []

    # Guarantee min_per_source where possible
    remaining = total
    allocated: dict[str, int] = {}
    for src in sources:
        n = min(min_per_source, len(by_src[src]), remaining)
        allocated[src] = n
        remaining -= n

    # Distribute remainder proportional to source size
    if remaining > 0:
        sizes = {s: len(by_src[s]) for s in sources}
        total_size = sum(sizes.values()) or 1
        for src in sources:
            extra = int(remaining * sizes[src] / total_size)
            room = len(by_src[src]) - allocated[src]
            take = min(extra, room)
            allocated[src] += take
            remaining -= take
        # leftovers
        for src in sources:
            if remaining <= 0:
                break
            room = len(by_src[src]) - allocated[src]
            take = min(room, remaining)
            allocated[src] += take
            remaining -= take

    chosen_ids: set[str] = set()
    for src in sources:
        pool = by_src[src][:]
        rng.shuffle(pool)
        for r in pool[: allocated[src]]:
            sample.append(r)
            chosen_ids.add(r["record_id"])
    return sample


def write_outputs(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    stats: FilterStats,
    sample: list[dict[str, Any]],
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    def dump(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(OUT / "accepted.jsonl", accepted)
    dump(OUT / "rejected.jsonl", rejected)
    # Train questions = accepted for now (specialized holdout happens later)
    dump(OUT / "train_questions.jsonl", [r for r in accepted if r.get("split") == "train"])
    dump(OUT / "manual_review_sample.jsonl", sample)

    # Manual review CSV template
    csv_path = OUT / "manual_review_template.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "record_id",
                "source",
                "language_ok",
                "kurmanji_ok",
                "punctuation_ok",
                "question_is_real",
                "context_ok",
                "encoding_ok",
                "license_metadata_ok",
                "accept",
                "comment",
            ]
        )
        for r in sample:
            w.writerow([r["record_id"], r["source"], "", "", "", "", "", "", "", "", ""])

    source_stats = {
        "raw_by_source": dict(stats.by_source_raw),
        "accepted_by_source": dict(stats.by_source_accepted),
        "rejected_reasons": dict(stats.reject_reasons),
    }
    (OUT / "source_statistics.json").write_text(
        json.dumps(source_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    qstats = {
        "question_types": dict(stats.question_types),
        "accepted_records": stats.accepted,
        "kmr_special_char_docs": stats.kmr_char_docs,
        "turkish_flagged_for_review": stats.turkish_flagged,
        "max_type_share": (
            max(stats.question_types.values()) / stats.accepted if stats.accepted else 0.0
        ),
    }
    (OUT / "question_statistics.json").write_text(
        json.dumps(qstats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dedup_report = {
        "raw_records": stats.raw_records,
        "exact_duplicates_removed": stats.exact_duplicates_removed,
        "punctuation_only_duplicates_removed": stats.punctuation_only_duplicates_removed,
        "near_duplicates_removed": stats.near_duplicates_removed,
        "accepted_records": stats.accepted,
        "rejected_records": stats.rejected,
    }
    (OUT / "deduplication_report.json").write_text(
        json.dumps(dedup_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # License manifest
    lic_path = OUT / "license_manifest.csv"
    with lic_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "source", "source_document_id", "source_url", "license"])
        for r in accepted:
            w.writerow(
                [
                    r["record_id"],
                    r["source"],
                    r.get("source_document_id"),
                    r.get("source_url"),
                    r.get("license"),
                ]
            )

    # SHA256 sums for key artifacts
    sums = []
    for name in [
        "accepted.jsonl",
        "rejected.jsonl",
        "train_questions.jsonl",
        "manual_review_sample.jsonl",
        "source_statistics.json",
        "question_statistics.json",
        "deduplication_report.json",
        "license_manifest.csv",
    ]:
        p = OUT / name
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            sums.append(f"{h}  {name}")
    (OUT / "DATASET_SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")

    readme = f"""# v2 question corpus (processed)

Generated: {date.today().isoformat()}

## Counts

- raw: {stats.raw_records}
- accepted: {stats.accepted}
- rejected: {stats.rejected}
- exact dupes removed: {stats.exact_duplicates_removed}
- punct-only dupes removed: {stats.punctuation_only_duplicates_removed}
- near dupes removed: {stats.near_duplicates_removed}

## Manual review

Fill `manual_review_template.csv` for the stratified sample (`manual_review_sample.jsonl`).
Source accept rate must be ≥ 90% before including that source in train.

## Next

1. Human review (Stage 3)
2. `scripts/check_v2_leakage.py` (Stage 4)
3. Specialized question-test holdout (Stage 5)
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sources",
        nargs="+",
        default=["tatoeba", "kurmanji_news", "kurdish_ai_corpus"],
    )
    p.add_argument("--manual-sample-size", type=int, default=400)
    p.add_argument("--min-per-source", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    records = load_raw_jsonls(args.sources)
    print(f"[info] loaded raw records: {len(records)}")
    accepted, rejected, stats = filter_corpus(records)
    sample = stratified_manual_sample(
        accepted,
        total=args.manual_sample_size,
        min_per_source=args.min_per_source,
        seed=args.seed,
    )
    write_outputs(accepted, rejected, stats, sample)

    report = {
        "raw_records": stats.raw_records,
        "accepted_records": stats.accepted,
        "rejected_records": stats.rejected,
        "exact_duplicates_removed": stats.exact_duplicates_removed,
        "punctuation_only_duplicates_removed": stats.punctuation_only_duplicates_removed,
        "near_duplicates_removed": stats.near_duplicates_removed,
        "reject_reasons": dict(stats.reject_reasons.most_common()),
        "accepted_by_source": dict(stats.by_source_accepted),
        "question_types": dict(stats.question_types),
        "manual_review_sample": len(sample),
        "max_question_type_share": (
            max(stats.question_types.values()) / stats.accepted if stats.accepted else 0.0
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if stats.accepted and report["max_question_type_share"] > 0.40:
        print(
            "[gate] WARNING: one question type exceeds 40% — diversify before train",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
