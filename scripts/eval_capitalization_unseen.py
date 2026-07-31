#!/usr/bin/env python3
"""Evaluate capitalization on hand-curated unseen-name / false-positive probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_capitalization.inference import CapitalizationRestorer  # noqa: E402
from kurmanji_capitalization.metrics import compute_capitalization_metrics  # noqa: E402
from kurmanji_capitalization.constants import IGNORE_LABEL  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("outputs/capitalization-xlm-r-base-v1/best"))
    p.add_argument(
        "--cases",
        type=Path,
        default=Path("data/test_capitalization_specialized/cases.jsonl"),
    )
    p.add_argument("--title-threshold", type=float, default=0.80)
    p.add_argument("--upper-threshold", type=float, default=0.90)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/capitalization-xlm-r-base-v1/unseen_name_eval.json"),
    )
    args = p.parse_args()

    cases = []
    with args.cases.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    restorer = CapitalizationRestorer(
        str(args.model),
        title_threshold=args.title_threshold,
        upper_threshold=args.upper_threshold,
    )

    y_true: list[str] = []
    y_pred: list[str] = []
    by_cat: dict[str, list[tuple[str, str]]] = {}
    errors = []

    for case in cases:
        scored = restorer.score_tokens(case["tokens"])
        cat = case.get("category", "other")
        by_cat.setdefault(cat, [])
        for gold, pred, tok in zip(case["labels"], scored, case["tokens"]):
            if gold == IGNORE_LABEL:
                continue
            y_true.append(gold)
            y_pred.append(pred["predicted_label"])
            by_cat[cat].append((gold, pred["predicted_label"]))
            if gold != pred["predicted_label"]:
                errors.append(
                    {
                        "id": case["id"],
                        "category": cat,
                        "token": tok,
                        "gold": gold,
                        "pred": pred["predicted_label"],
                        "confidence": pred["confidence"],
                        "note": case.get("note"),
                    }
                )

    metrics = compute_capitalization_metrics(y_true, y_pred)
    cat_stats = {}
    for cat, pairs in by_cat.items():
        if not pairs:
            continue
        yt, yp = zip(*pairs)
        cat_stats[cat] = compute_capitalization_metrics(list(yt), list(yp))["per_label"]

    # TITLE precision focus on common_word category (false positive rate)
    common = by_cat.get("common_word", [])
    fp_common = sum(1 for g, p in common if g == "KEEP" and p == "TITLE")

    report = {
        "n_cases": len(cases),
        "n_scored_tokens": len(y_true),
        "title_threshold": args.title_threshold,
        "upper_threshold": args.upper_threshold,
        "metrics": metrics,
        "per_category": cat_stats,
        "common_word_KEEP_to_TITLE_fp": fp_common,
        "common_word_n": len(common),
        "errors": errors[:50],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "n_scored_tokens": len(y_true),
        "TITLE": metrics["per_label"]["TITLE"],
        "UPPER": metrics["per_label"]["UPPER"],
        "KEEP": metrics["per_label"]["KEEP"],
        "macro_f1": metrics["capitalization_macro_f1"],
        "common_word_KEEP_to_TITLE_fp": fp_common,
        "n_errors": len(errors),
    }, indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
