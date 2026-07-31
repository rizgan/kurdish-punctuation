#!/usr/bin/env python3
"""
Honest long-text evaluation (checks sentence-boundary leakage).

Takes whole test articles (or multi-sentence chunks), strips , . ? !,
runs windowed inference, compares labels to gold — without using
sentence splits as example boundaries at eval time.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_punctuation.constants import (  # noqa: E402
    LABEL_TO_PUNCTUATION,
    LABELS,
    SENTENCE_BOUNDARY_LABELS,
)
from kurmanji_punctuation.dataset import (  # noqa: E402
    load_processed_jsonl,
    split_into_sentence_spans,
    tokens_and_labels_from_text,
)
from kurmanji_punctuation.inference import PunctuationRestorer  # noqa: E402
from kurmanji_punctuation.metrics import (  # noqa: E402
    compute_punctuation_metrics,
    format_classification_report,
)
from kurmanji_punctuation.normalization import normalize_for_dataset  # noqa: E402
from kurmanji_punctuation.text_utils import (  # noqa: E402
    join_words_with_punctuation,
    validate_text_preservation,
)


def load_article_ids_from_split(processed_jsonl: Path) -> set[str]:
    return {row["article_id"] for row in load_processed_jsonl(processed_jsonl)}


def load_articles(raw_jsonl: Path, keep_ids: set[str]) -> list[dict]:
    out = []
    with raw_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj["article_id"] in keep_ids:
                out.append(obj)
    return out


def gold_from_text(text: str, map_ellipsis: bool = True) -> tuple[list[str], list[str]] | None:
    text = normalize_for_dataset(text, map_ellipsis_to_period=map_ellipsis)
    return tokens_and_labels_from_text(text)


def make_chunks(
    text: str,
    *,
    mode: str,
    min_sents: int,
    max_sents: int,
    rng: random.Random,
) -> list[str]:
    text = normalize_for_dataset(text, map_ellipsis_to_period=True)
    if mode == "article":
        return [text] if text.strip() else []
    sents = split_into_sentence_spans(text)
    if not sents:
        return []
    chunks: list[str] = []
    i = 0
    while i < len(sents):
        n = rng.randint(min_sents, max_sents)
        piece = " ".join(sents[i : i + n]).strip()
        if piece:
            chunks.append(piece)
        i += n
    return chunks


def confusion_transitions(y_true: list[str], y_pred: list[str]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            c[f"{t} -> {p}"] += 1
    return dict(c.most_common())


def interesting_transitions(trans: dict[str, int]) -> dict[str, int]:
    keys = [
        "QUESTION -> PERIOD",
        "QUESTION -> O",
        "EXCLAMATION -> PERIOD",
        "COMMA -> O",
        "O -> COMMA",
        "PERIOD -> O",
        "O -> PERIOD",
        "PERIOD -> COMMA",
        "COMMA -> PERIOD",
    ]
    return {k: int(trans.get(k, 0)) for k in keys}


def evaluate_chunks(
    restorer: PunctuationRestorer,
    chunks: list[str],
    *,
    apply_thresholds: bool,
) -> dict:
    y_true: list[str] = []
    y_pred: list[str] = []
    n_ok = 0
    n_total = 0
    n_words = 0
    skipped = 0

    for chunk in chunks:
        gold = gold_from_text(chunk)
        if not gold:
            skipped += 1
            continue
        tokens, labels = gold
        if len(tokens) < 3:
            skipped += 1
            continue
        stripped = " ".join(tokens)  # no punctuation signal
        logits = restorer.mean_logits_for_words(tokens)
        probs = restorer.softmax_rows(logits)
        decoded = restorer.decode_probs(probs, apply_thresholds=apply_thresholds)
        pred_labels = [lab for lab, _ in decoded]
        if len(pred_labels) != len(labels):
            skipped += 1
            continue

        punct = [LABEL_TO_PUNCTUATION[lab] for lab in pred_labels]
        restored = join_words_with_punctuation(tokens, punct)
        n_total += 1
        n_words += len(tokens)
        if validate_text_preservation(stripped, restored):
            n_ok += 1

        y_true.extend(labels)
        y_pred.extend(pred_labels)

    metrics = compute_punctuation_metrics(y_true, y_pred) if y_true else {}
    trans = confusion_transitions(y_true, y_pred) if y_true else {}
    return {
        "n_chunks": len(chunks),
        "n_evaluated": n_total,
        "n_skipped": skipped,
        "n_words": n_words,
        "text_preservation_rate": (n_ok / n_total) if n_total else 0.0,
        "metrics": metrics,
        "punctuation_macro_f1_long_text": metrics.get("punctuation_macro_f1", 0.0),
        "sentence_boundary_f1": metrics.get("sentence_boundary_f1", 0.0),
        "transitions": interesting_transitions(trans),
        "transitions_top": dict(list(trans.items())[:30]),
        "label_support_gold": dict(Counter(y_true)),
        "label_support_pred": dict(Counter(y_pred)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("outputs/punctuation-xlm-r-base/best"))
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--raw", type=Path, default=Path("data/raw/wikipedia.jsonl"))
    p.add_argument("--test-split", type=Path, default=Path("data/processed/test.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/punctuation-xlm-r-base"))
    p.add_argument("--max-articles", type=int, default=400, help="Cap articles for runtime")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-thresholds", action="store_true", help="Argmax only (no confidence floors)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--tag", type=str, default="", help="Optional filename suffix, e.g. no_thr")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inf = cfg.get("inference", {})
    rng = random.Random(args.seed)

    test_ids = load_article_ids_from_split(args.test_split)
    articles = load_articles(args.raw, test_ids)
    rng.shuffle(articles)
    articles = articles[: args.max_articles]
    print(f"Test article_ids available={len(test_ids)} using={len(articles)}")

    restorer = PunctuationRestorer(
        model_path=str(args.model),
        device=args.device,
        max_length=int(inf.get("max_length", 256)),
        overlap_words=int(inf.get("overlap_words", 32)),
        batch_size=int(inf.get("batch_size", 16)),
        minimum_confidence=inf.get("minimum_confidence"),
    )

    apply_thr = not args.no_thresholds
    report: dict = {
        "model": str(args.model),
        "n_articles": len(articles),
        "apply_thresholds": apply_thr,
        "label_counts_from_statistics": json.loads(
            Path("data/processed/statistics.json").read_text(encoding="utf-8")
        ).get("label_counts"),
        "modes": {},
    }

    # Mode A: whole articles
    article_chunks = []
    for art in articles:
        article_chunks.extend(make_chunks(art["text"], mode="article", min_sents=1, max_sents=1, rng=rng))
    print(f"Evaluating whole-article mode: {len(article_chunks)} texts …")
    report["modes"]["article"] = evaluate_chunks(restorer, article_chunks, apply_thresholds=apply_thr)

    # Mode B: 3–10 sentence bundles (no single-sentence examples)
    multi_chunks = []
    for art in articles:
        multi_chunks.extend(
            make_chunks(art["text"], mode="multi", min_sents=3, max_sents=10, rng=rng)
        )
    print(f"Evaluating multi-sentence mode (3-10): {len(multi_chunks)} chunks …")
    report["modes"]["multi_sentence_3_10"] = evaluate_chunks(
        restorer, multi_chunks, apply_thresholds=apply_thr
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = args.output_dir / f"long_text_eval{suffix}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable summary
    lines = ["=== Honest long-text evaluation ===", ""]
    for mode_name, block in report["modes"].items():
        m = block["metrics"]
        lines.append(f"## {mode_name}")
        lines.append(f"chunks_evaluated: {block['n_evaluated']}  words: {block['n_words']}")
        lines.append(f"punctuation_macro_f1_long_text: {block['punctuation_macro_f1_long_text']:.4f}")
        lines.append(f"sentence_boundary_f1: {block['sentence_boundary_f1']:.4f}")
        lines.append(f"text_preservation_rate: {block['text_preservation_rate']:.4f}")
        if m:
            lines.append(format_classification_report(m).rstrip())
        lines.append("key transitions:")
        for k, v in block["transitions"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    summary_path = args.output_dir / f"long_text_eval{suffix}.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {out_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
