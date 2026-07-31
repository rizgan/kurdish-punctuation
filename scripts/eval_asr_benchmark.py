#!/usr/bin/env python3
"""Evaluate punctuation model on ASR benchmark (preserve ASR word forms)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_punctuation.dataset import tokens_and_labels_from_text  # noqa: E402
from kurmanji_punctuation.inference import PunctuationRestorer  # noqa: E402
from kurmanji_punctuation.metrics import compute_punctuation_metrics  # noqa: E402
from kurmanji_punctuation.normalization import normalize_for_inference  # noqa: E402
from kurmanji_punctuation.text_utils import (  # noqa: E402
    extract_words,
    validate_text_preservation,
)


def load_jsonl(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists() or path.stat().st_size == 0:
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[str(obj["id"])] = str(obj["text"])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asr", type=Path, default=Path("data/asr_benchmark_v1/asr_raw.jsonl"))
    p.add_argument("--gold", type=Path, default=Path("data/asr_benchmark_v1/punctuation_gold.jsonl"))
    p.add_argument(
        "--punctuation-model",
        type=Path,
        default=Path("models/punctuation/kurmanji-xlm-r-base-v2"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/asr_benchmark_v1/punctuation_eval.json"),
    )
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    asr = load_jsonl(args.asr)
    gold = load_jsonl(args.gold)
    if not asr or not gold:
        print(
            "ASR benchmark is empty. Fill asr_raw.jsonl and punctuation_gold.jsonl "
            "(see data/asr_benchmark_v1/README.md).",
            file=sys.stderr,
        )
        return 2

    ids = sorted(set(asr) & set(gold))
    missing_gold = sorted(set(asr) - set(gold))
    missing_asr = sorted(set(gold) - set(asr))

    restorer = PunctuationRestorer(str(args.punctuation_model), device=args.device)

    y_true: list[str] = []
    y_pred: list[str] = []
    preserve_ok = 0
    preserve_fail = 0
    word_mismatch = 0
    examples = []

    for i, rid in enumerate(ids):
        raw = normalize_for_inference(asr[rid])
        gtext = normalize_for_inference(gold[rid])
        pred = restorer.restore(raw)

        if validate_text_preservation(raw, pred):
            preserve_ok += 1
        else:
            preserve_fail += 1

        if extract_words(raw) != extract_words(gtext):
            word_mismatch += 1
            continue

        gold_pair = tokens_and_labels_from_text(gtext)
        pred_pair = tokens_and_labels_from_text(pred)
        if not gold_pair or not pred_pair:
            continue
        g_toks, g_labs = gold_pair
        p_toks, p_labs = pred_pair
        if len(g_labs) != len(p_labs) or g_toks != p_toks:
            # Same words expected; if punct model re-tokenizes oddly, skip
            word_mismatch += 1
            continue
        y_true.extend(g_labs)
        y_pred.extend(p_labs)

        if len(examples) < 10 and g_labs != p_labs:
            examples.append({"id": rid, "asr": raw, "gold": gtext, "pred": pred})

    if not y_true:
        print("No aligned examples to score.", file=sys.stderr)
        return 1

    metrics = compute_punctuation_metrics(y_true, y_pred)
    report = {
        "n_ids_overlap": len(ids),
        "missing_gold": missing_gold[:20],
        "missing_asr": missing_asr[:20],
        "word_mismatch_skipped": word_mismatch,
        "text_preservation_ok": preserve_ok,
        "text_preservation_fail": preserve_fail,
        "text_preservation": preserve_ok / max(1, preserve_ok + preserve_fail),
        "n_scored_tokens": len(y_true),
        "PERIOD_f1": metrics["per_label"].get("PERIOD", {}).get("f1"),
        "COMMA_f1": metrics["per_label"].get("COMMA", {}).get("f1"),
        "QUESTION_f1": metrics["per_label"].get("QUESTION", {}).get("f1"),
        "sentence_boundary_f1": metrics["sentence_boundary_f1"],
        "metrics": metrics,
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in report if k not in ("metrics", "examples", "missing_gold", "missing_asr")},
            indent=2,
        )
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
