#!/usr/bin/env python3
"""Score filled human-audit CSV and compare to assisted review (offline).

Emits a machine-readable report with:
  overall_accept_rate, language_accuracy, question_validity_rate,
  punctuation_accuracy, context_accuracy, accept_rate_by_source,
  agreement_with_assisted_review, reviewed_records, invalid_or_missing_rows

Final status: PASS | PASS_WITH_NOTES | FAIL
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2_question_processed"

REQUIRED_FIELDS = (
    "record_id",
    "source",
    "kurmanji_ok",
    "question_is_real",
    "punctuation_ok",
    "context_ok",
    "accept",
)


def as_bool(v: str | bool | None) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"", "na", "n/a"}:
        return None
    if s in {"1", "true", "yes", "y", "accept", "ok"}:
        return True
    if s in {"0", "false", "no", "n", "reject"}:
        return False
    return None


def rate(xs: list[bool]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def decide_status(
    *,
    overall: float | None,
    language: float | None,
    question: float | None,
    by_source: dict[str, dict],
    systematic_language_mix: bool,
    reviewed: int,
    min_reviewed: int = 400,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    if reviewed < min_reviewed:
        notes.append(f"reviewed_records={reviewed} < {min_reviewed}")
    if systematic_language_mix:
        return "FAIL", notes + ["systematic language mixing flagged"]

    o = overall if overall is not None else 0.0
    lang = language if language is not None else 0.0
    qv = question if question is not None else 0.0

    if o < 0.85:
        return "FAIL", notes + [f"overall_accept_rate={o:.4f} < 0.85"]

    # Localized source failures (for PASS_WITH_NOTES)
    weak_sources = [
        s
        for s, st in by_source.items()
        if st["n"] >= 20 and (st["accept_rate"] or 0) < 0.85
    ]
    if weak_sources:
        notes.append("weak_sources=" + ",".join(weak_sources))

    if o >= 0.90 and lang >= 0.95 and qv >= 0.95 and not weak_sources:
        if notes:
            # e.g. sample slightly under 400 but metrics strong
            return "PASS_WITH_NOTES", notes
        return "PASS", notes

    if o >= 0.85 and (weak_sources or o < 0.90 or lang < 0.95 or qv < 0.95):
        if lang < 0.95:
            notes.append(f"language_accuracy={lang:.4f} < 0.95")
        if qv < 0.95:
            notes.append(f"question_validity_rate={qv:.4f} < 0.95")
        if o < 0.90:
            notes.append(f"overall_accept_rate={o:.4f} < 0.90")
        # PASS_WITH_NOTES only if errors look localized (1–2 sources) or mild shortfall
        if len(weak_sources) <= 2 or (o >= 0.85 and not systematic_language_mix):
            return "PASS_WITH_NOTES", notes

    return "FAIL", notes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument(
        "--assisted",
        type=Path,
        default=OUT / "manual_review_assisted.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=OUT / "human_audit_report.json",
    )
    p.add_argument(
        "--min-reviewed",
        type=int,
        default=400,
        help="Preferred minimum reviewed rows (warning / notes if lower)",
    )
    p.add_argument(
        "--flag-language-mix",
        action="store_true",
        help="Force FAIL for systematic language mixing (set by auditor)",
    )
    args = p.parse_args()

    with args.csv.open(encoding="utf-8", newline="") as f:
        human = list(csv.DictReader(f))

    assisted_map: dict[str, bool] = {}
    if args.assisted.exists():
        with args.assisted.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                b = as_bool(row.get("accept"))
                if b is not None:
                    assisted_map[row["record_id"]] = b

    by_src_accept: dict[str, list[bool]] = defaultdict(list)
    q_real: list[bool] = []
    punct_ok: list[bool] = []
    lang_ok: list[bool] = []
    context_ok: list[bool] = []
    accepts: list[bool] = []
    invalid_or_missing = 0
    agree = 0
    agree_n = 0

    for row in human:
        missing_required = any(
            not str(row.get(k, "")).strip() for k in ("record_id", "source", "accept")
        )
        a = as_bool(row.get("accept"))
        soft = {
            "kurmanji_ok": as_bool(row.get("kurmanji_ok")),
            "question_is_real": as_bool(row.get("question_is_real")),
            "punctuation_ok": as_bool(row.get("punctuation_ok")),
            "context_ok": as_bool(row.get("context_ok")),
        }
        if (
            missing_required
            or a is None
            or soft["kurmanji_ok"] is None
            or soft["question_is_real"] is None
        ):
            invalid_or_missing += 1
            continue

        accepts.append(a)
        by_src_accept[row.get("source", "?")].append(a)
        lang_ok.append(soft["kurmanji_ok"])
        q_real.append(soft["question_is_real"])
        if soft["punctuation_ok"] is not None:
            punct_ok.append(soft["punctuation_ok"])
        if soft["context_ok"] is not None:
            context_ok.append(soft["context_ok"])

        rid = row["record_id"]
        if rid in assisted_map:
            agree_n += 1
            if assisted_map[rid] == a:
                agree += 1

    accept_rate_by_source = {
        s: {"n": len(v), "accepted": sum(v), "accept_rate": rate(v)}
        for s, v in sorted(by_src_accept.items())
    }

    overall = rate(accepts)
    language = rate(lang_ok)
    question = rate(q_real)
    punctuation = rate(punct_ok)
    context = rate(context_ok)

    status, notes = decide_status(
        overall=overall,
        language=language,
        question=question,
        by_source=accept_rate_by_source,
        systematic_language_mix=bool(args.flag_language_mix),
        reviewed=len(accepts),
        min_reviewed=args.min_reviewed,
    )

    report = {
        "scored_at": date.today().isoformat(),
        "csv": str(args.csv),
        "status": status,
        "notes": notes,
        "reviewed_records": len(accepts),
        "invalid_or_missing_rows": invalid_or_missing,
        "n_rows_in_csv": len(human),
        "overall_accept_rate": overall,
        "language_accuracy": language,
        "question_validity_rate": question,
        "punctuation_accuracy": punctuation,
        "context_accuracy": context,
        "accept_rate_by_source": accept_rate_by_source,
        "agreement_with_assisted_review": (agree / agree_n) if agree_n else None,
        "agreement_n": agree_n,
        "gates": {
            "PASS": {
                "overall_accept_rate_ge_90": (overall or 0) >= 0.90,
                "language_accuracy_ge_95": (language or 0) >= 0.95,
                "question_validity_rate_ge_95": (question or 0) >= 0.95,
            },
            "PASS_WITH_NOTES": {
                "overall_accept_rate_ge_85": (overall or 0) >= 0.85,
                "errors_localized_1_or_2_sources": len(
                    [
                        s
                        for s, st in accept_rate_by_source.items()
                        if st["n"] >= 20 and (st["accept_rate"] or 0) < 0.85
                    ]
                )
                <= 2,
            },
            "FAIL": {
                "overall_accept_rate_lt_85": (overall or 0) < 0.85,
                "systematic_language_mix": bool(args.flag_language_mix),
            },
        },
        "corpus_review_complete": status == "PASS",
        "release_note": (
            "Frozen research release remains valid; "
            "update model_card corpus_review_status after this report."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[gate] {status}")
    if status == "FAIL":
        return 2
    if status == "PASS_WITH_NOTES":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
