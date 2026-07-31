#!/usr/bin/env python3
"""Honest long-text capitalization evaluation on full Wikipedia articles.

original (cased) → lower → sentence-start rule → windowed CapitalizationRestorer
→ compare labels to gold (IGNORE / sentence-starts excluded from F1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_capitalization.casing import (  # noqa: E402
    is_letter,
    kurmanji_lower,
    kurmanji_title_token,
)
from kurmanji_capitalization.constants import IGNORE_LABEL  # noqa: E402
from kurmanji_capitalization.dataset import (  # noqa: E402
    load_processed_jsonl,
    tokens_and_labels_from_article,
)
from kurmanji_capitalization.inference import CapitalizationRestorer  # noqa: E402
from kurmanji_capitalization.metrics import compute_capitalization_metrics  # noqa: E402
from kurmanji_capitalization.normalization import normalize_for_dataset  # noqa: E402
from kurmanji_capitalization.sentence_rule import capitalize_sentence_starts  # noqa: E402
from kurmanji_capitalization.text_utils import (  # noqa: E402
    tokenize_words_and_punct,
    validate_case_only_transformation,
)


def load_article_ids(processed: Path) -> set[str]:
    return {row["article_id"] for row in load_processed_jsonl(processed)}


def load_articles(raw: Path, keep: set[str]) -> list[dict]:
    out = []
    with raw.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj["article_id"] in keep:
                out.append(obj)
    return out


def first_letter_is_upper(token: str) -> bool:
    for ch in token:
        if is_letter(ch):
            return ch == kurmanji_title_token(kurmanji_lower(ch)) or ch.isupper()
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("outputs/capitalization-xlm-r-base-v1/best"))
    p.add_argument("--config", type=Path, default=Path("configs/capitalization-v1.yaml"))
    p.add_argument("--raw", type=Path, default=Path("data/raw/wikipedia.jsonl"))
    p.add_argument(
        "--processed-split",
        type=Path,
        default=Path("data/processed_capitalization/test.jsonl"),
    )
    p.add_argument("--max-articles", type=int, default=400)
    p.add_argument("--title-threshold", type=float, default=None)
    p.add_argument("--upper-threshold", type=float, default=None)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/capitalization-xlm-r-base-v1/long_text_eval.json"),
    )
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inf = cfg.get("inference", {})
    conf = inf.get("minimum_confidence", {})
    title_thr = (
        args.title_threshold if args.title_threshold is not None else float(conf.get("TITLE", 0.80))
    )
    upper_thr = (
        args.upper_threshold if args.upper_threshold is not None else float(conf.get("UPPER", 0.90))
    )

    keep_ids = load_article_ids(args.processed_split)
    articles = load_articles(args.raw, keep_ids)[: args.max_articles]

    restorer = CapitalizationRestorer(
        str(args.model),
        max_length=int(inf.get("max_length", 256)),
        overlap_words=int(inf.get("overlap_words", 32)),
        title_threshold=title_thr,
        upper_threshold=upper_thr,
    )

    y_true: list[str] = []
    y_pred: list[str] = []
    n_preserve_ok = 0
    n_preserve_fail = 0
    n_sent_ok = 0
    n_sent_total = 0
    n_articles_ok = 0
    n_input_not_lower = 0
    n_trainable_checked = 0

    for i, art in enumerate(articles):
        text = normalize_for_dataset(art["text"])
        triple = tokens_and_labels_from_article(text)
        if not triple:
            continue
        input_tokens, gold_labels, _orig = triple
        after_rule = capitalize_sentence_starts(kurmanji_lower(text))
        if tokenize_words_and_punct(after_rule) != input_tokens:
            continue

        preds = restorer.score_tokens(input_tokens)
        if len(preds) != len(gold_labels):
            continue

        restored = restorer.restore(text)
        if validate_case_only_transformation(after_rule, restored):
            n_preserve_ok += 1
        else:
            n_preserve_fail += 1

        for tok, gold, pred in zip(input_tokens, gold_labels, preds):
            if pred["sentence_start"]:
                n_sent_total += 1
                if first_letter_is_upper(pred["token_after"]):
                    n_sent_ok += 1
                continue
            if gold == IGNORE_LABEL:
                continue

            n_trainable_checked += 1
            if tok != kurmanji_lower(tok):
                n_input_not_lower += 1

            y_true.append(gold)
            y_pred.append(pred["predicted_label"])

        n_articles_ok += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(articles)} articles", flush=True)

    metrics = compute_capitalization_metrics(y_true, y_pred)
    gate = {
        "TITLE_f1": metrics["per_label"]["TITLE"]["f1"],
        "TITLE_precision": metrics["per_label"]["TITLE"]["precision"],
        "TITLE_recall": metrics["per_label"]["TITLE"]["recall"],
        "UPPER_f1": metrics["per_label"]["UPPER"]["f1"],
        "KEEP_f1": metrics["per_label"]["KEEP"]["f1"],
        "macro_f1": metrics["capitalization_macro_f1"],
        "preservation_rate": n_preserve_ok / max(1, n_preserve_ok + n_preserve_fail),
        "pass_TITLE_f1": metrics["per_label"]["TITLE"]["f1"] >= 0.85,
        "pass_TITLE_precision": metrics["per_label"]["TITLE"]["precision"] >= 0.88,
        "pass_UPPER_f1": metrics["per_label"]["UPPER"]["f1"] >= 0.85,
        "pass_KEEP_f1": metrics["per_label"]["KEEP"]["f1"] >= 0.97,
        "pass_preservation": n_preserve_fail == 0,
    }
    gate["v1_freeze_candidate"] = all(
        gate[k]
        for k in (
            "pass_TITLE_f1",
            "pass_TITLE_precision",
            "pass_UPPER_f1",
            "pass_KEEP_f1",
            "pass_preservation",
        )
    )

    report = {
        "n_articles_requested": len(articles),
        "n_articles_scored": n_articles_ok,
        "n_scored_tokens": len(y_true),
        "title_threshold": title_thr,
        "upper_threshold": upper_thr,
        "case_only_preservation_ok": n_preserve_ok,
        "case_only_preservation_fail": n_preserve_fail,
        "sentence_start_accuracy": n_sent_ok / max(1, n_sent_total),
        "sentence_start_n": n_sent_total,
        "trainable_input_not_lower": n_input_not_lower,
        "trainable_input_checked": n_trainable_checked,
        "input_casing_ok": n_input_not_lower == 0,
        "gate": gate,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gate": gate, "n_articles_scored": n_articles_ok, "n_tokens": len(y_true)}, indent=2))
    print(json.dumps(metrics["per_label"], indent=2, ensure_ascii=False))
    print(f"macro_f1={metrics['capitalization_macro_f1']:.4f} wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
