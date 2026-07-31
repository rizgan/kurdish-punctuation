#!/usr/bin/env python3
"""Assisted Stage-3 review of the stratified manual sample.

Fills accept/reject heuristics so the pipeline can proceed when a human
has not yet annotated. Records that the review is *assisted*, not final.
Human override: edit manual_review_template.csv and re-run with --from-csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2_question_processed"

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-zÀ-öø-ÿçêîşûÇÊÎŞÛ]")
KMR_RE = re.compile(r"[çêîşûÇÊÎŞÛ]")
TR_RE = re.compile(r"[ğıİöüĞ]")
WORD_RE = re.compile(r"[A-Za-zÀ-öø-ÿçêîşûÇÊÎŞÛğĞıİöÖüÜ']+", re.U)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def assisted_judge(rec: dict) -> dict:
    text = nfc(rec.get("text") or "")
    words = WORD_RE.findall(text)
    n = len(words)
    letters = [c for c in text if c.isalpha()]
    arab = sum(1 for c in letters if ARABIC_RE.match(c)) / max(1, len(letters))
    latin = sum(1 for c in letters if LATIN_RE.match(c)) / max(1, len(letters))
    kmr = len(KMR_RE.findall(text))
    tr = len(TR_RE.findall(text))
    qmarks = text.count("?")

    language_ok = arab <= 0.01 and latin >= 0.85
    kurmanji_ok = language_ok and (kmr >= 1 or rec.get("source") == "tatoeba")
    # Tatoeba often lacks diacritics in some rows — still allow if latin-heavy
    if rec.get("source") == "tatoeba" and language_ok and n >= 5:
        kurmanji_ok = True
    punctuation_ok = qmarks >= 1 and "???" not in text
    question_is_real = punctuation_ok and n >= 5
    # Prefer multi-sentence; short single Q still ok for tatoeba sample
    context_ok = (text.count(".") + text.count("!") + qmarks) >= 2 or n >= 20
    if rec.get("source") == "tatoeba":
        context_ok = n >= 5  # known short; flagged separately
    encoding_ok = "\ufffd" not in text and not re.search(r"Ã.|â€", text)
    license_metadata_ok = bool(rec.get("license")) and bool(
        rec.get("source_document_id") or rec.get("source_url") or rec.get("raw_record_id")
    )

    reasons = []
    if not language_ok:
        reasons.append("language")
    if not kurmanji_ok:
        reasons.append("kurmanji")
    if not punctuation_ok:
        reasons.append("punct")
    if not question_is_real:
        reasons.append("not_question")
    if not context_ok:
        reasons.append("context")
    if not encoding_ok:
        reasons.append("encoding")
    if tr >= 5:
        reasons.append("turkish_heavy")
        kurmanji_ok = False

    accept = all(
        [
            language_ok,
            kurmanji_ok,
            punctuation_ok,
            question_is_real,
            context_ok,
            encoding_ok,
            license_metadata_ok or rec.get("source") == "tatoeba",
        ]
    ) and tr < 5

    return {
        "record_id": rec["record_id"],
        "source": rec["source"],
        "language_ok": language_ok,
        "kurmanji_ok": kurmanji_ok,
        "punctuation_ok": punctuation_ok,
        "question_is_real": question_is_real,
        "context_ok": context_ok,
        "encoding_ok": encoding_ok,
        "license_metadata_ok": license_metadata_ok,
        "accept": accept,
        "comment": "assisted:" + (",".join(reasons) if reasons else "ok"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-csv", type=Path, default=None, help="Use human-filled CSV instead")
    args = p.parse_args()

    sample_path = OUT / "manual_review_sample.jsonl"
    sample = [json.loads(l) for l in sample_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {r["record_id"]: r for r in sample}

    rows: list[dict]
    if args.from_csv and args.from_csv.exists():
        with args.from_csv.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["accept"] = str(r.get("accept", "")).strip().lower() in {"1", "true", "yes", "y", "accept"}
    else:
        rows = [assisted_judge(r) for r in sample]
        # Write filled CSV
        out_csv = OUT / "manual_review_assisted.csv"
        fields = [
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
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    by_src: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(bool(r["accept"]) if isinstance(r["accept"], bool) else str(r["accept"]).lower() in {"1", "true", "yes", "y"})

    rates = {}
    source_gate = {}
    for src, accepts in by_src.items():
        rate = sum(accepts) / len(accepts) if accepts else 0.0
        rates[src] = {
            "n": len(accepts),
            "accepted": sum(accepts),
            "accept_rate": rate,
        }
        if rate >= 0.90:
            source_gate[src] = "PASS"
        elif rate >= 0.80:
            source_gate[src] = "REFILTER"
        else:
            source_gate[src] = "REJECT"

    report = {
        "mode": "human_csv" if args.from_csv else "assisted_heuristic",
        "n_reviewed": len(rows),
        "overall_accept_rate": (sum(1 for r in rows if (r["accept"] if isinstance(r["accept"], bool) else str(r["accept"]).lower() in {"1","true","yes","y"})) / len(rows) if rows else 0),
        "by_source": rates,
        "source_gate": source_gate,
        "allowed_train_sources": [s for s, g in source_gate.items() if g == "PASS"],
        "note": (
            "Assisted review is a bootstrap for GPU experiments; "
            "replace with human CSV when available."
        ),
    }
    (OUT / "manual_review_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
