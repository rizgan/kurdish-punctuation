#!/usr/bin/env python3
"""Evaluate a checkpoint on specialized question-test contexts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_punctuation.constants import LABEL_TO_PUNCTUATION  # noqa: E402
from kurmanji_punctuation.dataset import tokens_and_labels_from_text  # noqa: E402
from kurmanji_punctuation.inference import PunctuationRestorer  # noqa: E402
from kurmanji_punctuation.metrics import compute_punctuation_metrics  # noqa: E402
from kurmanji_punctuation.normalization import normalize_for_dataset  # noqa: E402
from kurmanji_punctuation.text_utils import (  # noqa: E402
    join_words_with_punctuation,
    validate_text_preservation,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/v2-exp-01.effective.yaml"))
    p.add_argument(
        "--data",
        type=Path,
        default=Path("data/test_question_specialized/contexts.jsonl"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-thresholds", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inf = cfg.get("inference", {})
    restorer = PunctuationRestorer(
        model_path=str(args.model),
        max_length=int(inf.get("max_length", 256)),
        overlap_words=int(inf.get("overlap_words", 32)),
        batch_size=int(inf.get("batch_size", 16)),
        minimum_confidence=inf.get("minimum_confidence"),
    )
    apply_thr = not args.no_thresholds

    y_true: list[str] = []
    y_pred: list[str] = []
    n_ok = n_total = 0
    by_source: dict[str, dict] = {}
    errors = []

    rows = [json.loads(l) for l in args.data.read_text(encoding="utf-8").splitlines() if l.strip()]
    for row in rows:
        text = normalize_for_dataset(row["text"], map_ellipsis_to_period=True)
        gold = tokens_and_labels_from_text(text)
        if not gold:
            continue
        tokens, labels = gold
        if len(tokens) < 3:
            continue
        logits = restorer.mean_logits_for_words(tokens)
        probs = restorer.softmax_rows(logits)
        decoded = restorer.decode_probs(probs, apply_thresholds=apply_thr)
        pred = [lab for lab, _ in decoded]
        if len(pred) != len(labels):
            continue
        punct = [LABEL_TO_PUNCTUATION[lab] for lab in pred]
        restored = join_words_with_punctuation(tokens, punct)
        stripped = " ".join(tokens)
        n_total += 1
        if validate_text_preservation(stripped, restored):
            n_ok += 1
        y_true.extend(labels)
        y_pred.extend(pred)
        src = row.get("source", "unknown")
        by_source.setdefault(src, {"y_true": [], "y_pred": []})
        by_source[src]["y_true"].extend(labels)
        by_source[src]["y_pred"].extend(pred)
        for i, (t, pr) in enumerate(zip(labels, pred)):
            if t == "QUESTION" or pr == "QUESTION":
                if t != pr:
                    left = " ".join(tokens[max(0, i - 20) : i])
                    right = " ".join(tokens[i + 1 : i + 21])
                    errors.append(
                        {
                            "record_id": row.get("record_id"),
                            "token_index": i,
                            "context": f"{left} [{tokens[i]}] {right}",
                            "gold": t,
                            "predicted": pr,
                            "error_type": f"{t}_TO_{pr}",
                            "source": src,
                        }
                    )

    metrics = compute_punctuation_metrics(y_true, y_pred)
    src_metrics = {
        s: compute_punctuation_metrics(v["y_true"], v["y_pred"]) for s, v in by_source.items()
    }
    report = {
        "n_contexts": len(rows),
        "n_evaluated": n_total,
        "text_preservation": (n_ok / n_total) if n_total else 0.0,
        "metrics": metrics,
        "by_source": src_metrics,
        "question_support_gold": Counter(y_true).get("QUESTION", 0),
        "question_support_pred": Counter(y_pred).get("QUESTION", 0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    err_path = args.output.with_name("question_errors_specialized.jsonl")
    with err_path.open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(json.dumps({
        "QUESTION": metrics.get("per_label", {}).get("QUESTION") if isinstance(metrics.get("per_label"), dict) else None,
        "macro": metrics.get("punctuation_macro_f1"),
        "boundary": metrics.get("sentence_boundary_f1"),
        "preservation": report["text_preservation"],
        "n": n_total,
    }, indent=2, default=str))
    # Also print F1 keys if nested differently
    q = {k: v for k, v in metrics.items() if "QUESTION" in k or k.endswith("_QUESTION")}
    print("question_keys", q)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
