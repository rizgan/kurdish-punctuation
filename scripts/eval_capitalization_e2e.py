#!/usr/bin/env python3
"""End-to-end: lower+no-punct → punctuation v2 → capitalization v1.

Reports capitalization metrics on gold punctuation vs predicted punctuation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_capitalization.casing import kurmanji_lower  # noqa: E402
from kurmanji_capitalization.constants import IGNORE_LABEL, SUPPORTED_PUNCT  # noqa: E402
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
from kurmanji_punctuation.inference import PunctuationRestorer  # noqa: E402
from kurmanji_punctuation.text_utils import strip_supported_punctuation  # noqa: E402


def load_articles(raw: Path, keep: set[str], limit: int) -> list[dict]:
    out = []
    with raw.open(encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if obj["article_id"] in keep:
                out.append(obj)
            if len(out) >= limit:
                break
    return out


def align_cap_metrics(gold_text: str, pred_text: str) -> tuple[list[str], list[str]] | None:
    """Compare capitalization labels on predicted text vs gold article labels by word index.

    Uses gold labels from gold_text; predicts labels from pred_text tokens when lengths match
    after stripping to letter-words... Actually better: derive gold from gold_text, run
    restorer scoring on pred_text's after-rule tokens only if token lower-forms align.
    """
    gold_trip = tokens_and_labels_from_article(gold_text)
    if not gold_trip:
        return None
    g_in, g_lab, _ = gold_trip
    p_trip = tokens_and_labels_from_article(pred_text)
    # For predicted punctuation path, gold casing is still from original;
    # we score the capitalization model on pred_text (which has predicted punct).
    return None  # filled in main differently


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--punctuation-model",
        type=Path,
        default=Path("models/punctuation/kurmanji-xlm-r-base-v2"),
    )
    p.add_argument(
        "--capitalization-model",
        type=Path,
        default=Path("outputs/capitalization-xlm-r-base-v1/best"),
    )
    p.add_argument("--raw", type=Path, default=Path("data/raw/wikipedia.jsonl"))
    p.add_argument(
        "--processed-split",
        type=Path,
        default=Path("data/processed_capitalization/test.jsonl"),
    )
    p.add_argument("--max-articles", type=int, default=200)
    p.add_argument("--title-threshold", type=float, default=0.80)
    p.add_argument("--upper-threshold", type=float, default=0.90)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/capitalization-xlm-r-base-v1/e2e_pipeline_eval.json"),
    )
    args = p.parse_args()

    keep = {r["article_id"] for r in load_processed_jsonl(args.processed_split)}
    articles = load_articles(args.raw, keep, args.max_articles)

    punct = PunctuationRestorer(str(args.punctuation_model))
    cap = CapitalizationRestorer(
        str(args.capitalization_model),
        title_threshold=args.title_threshold,
        upper_threshold=args.upper_threshold,
    )

    gold_true: list[str] = []
    gold_pred: list[str] = []
    pipe_true: list[str] = []
    pipe_pred: list[str] = []
    preserve_fail = 0
    examples = []

    for i, art in enumerate(articles):
        text = normalize_for_dataset(art["text"])
        triple = tokens_and_labels_from_article(text)
        if not triple:
            continue
        in_toks, gold_labs, _ = triple

        # A) capitalization on gold punctuation
        scored_gold = cap.score_tokens(in_toks)
        for g, s in zip(gold_labs, scored_gold):
            if g == IGNORE_LABEL or s["sentence_start"]:
                continue
            gold_true.append(g)
            gold_pred.append(s["predicted_label"])

        # B) full pipeline: lower + strip punct → punct model → cap model
        bare = kurmanji_lower(strip_supported_punctuation(text))
        # Wiki leftovers like thumb|... break punctuation preservation; neutralize.
        bare = bare.replace("|", " ").replace("/", " ")
        bare = " ".join(bare.split())
        try:
            punctuated = punct.restore(bare)
        except Exception as exc:
            if len(examples) < 8:
                examples.append(
                    {
                        "article_id": art["article_id"],
                        "error": f"punctuation_restore_failed: {type(exc).__name__}",
                    }
                )
            continue
        after_rule = capitalize_sentence_starts(kurmanji_lower(punctuated))
        try:
            final = cap.restore(punctuated)
        except Exception as exc:
            preserve_fail += 1
            if len(examples) < 8:
                examples.append(
                    {
                        "article_id": art["article_id"],
                        "error": f"capitalization_restore_failed: {type(exc).__name__}",
                    }
                )
            continue
        if not validate_case_only_transformation(after_rule, final):
            preserve_fail += 1

        # Align on lowercased word sequence (ignore punct tokens)
        def words_lower(t: str) -> list[str]:
            return [
                kurmanji_lower(x)
                for x in tokenize_words_and_punct(t)
                if x not in SUPPORTED_PUNCT
            ]

        gold_words = words_lower(text)
        final_words = words_lower(final)
        # Build gold labels for words only
        gold_word_labs = []
        for tok, lab in zip(in_toks, gold_labs):
            if tok in SUPPORTED_PUNCT:
                continue
            gold_word_labs.append(lab)

        if len(gold_words) == len(final_words) == len(gold_word_labs):
            final_scored = cap.score_tokens(tokenize_words_and_punct(after_rule))
            word_pairs = [
                (s["predicted_label"], s["sentence_start"])
                for s in final_scored
                if s["token_after_rule"] not in SUPPORTED_PUNCT
            ]
            if len(word_pairs) == len(gold_word_labs):
                for g, (pr, ss) in zip(gold_word_labs, word_pairs):
                    if g == IGNORE_LABEL or ss:
                        continue
                    pipe_true.append(g)
                    pipe_pred.append(pr)

        if len(examples) < 5:
            examples.append(
                {
                    "article_id": art["article_id"],
                    "bare": bare[:180],
                    "punctuated": punctuated[:180],
                    "final": final[:180],
                }
            )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(articles)}", flush=True)

    report = {
        "n_articles": len(articles),
        "title_threshold": args.title_threshold,
        "upper_threshold": args.upper_threshold,
        "capitalization_on_gold_punctuation": compute_capitalization_metrics(gold_true, gold_pred),
        "capitalization_on_predicted_punctuation": compute_capitalization_metrics(pipe_true, pipe_pred)
        if pipe_true
        else None,
        "n_tokens_gold_punct": len(gold_true),
        "n_tokens_pred_punct": len(pipe_true),
        "case_only_preservation_fail": preserve_fail,
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "gold_punct_macro": report["capitalization_on_gold_punctuation"]["capitalization_macro_f1"],
        "gold_punct_TITLE": report["capitalization_on_gold_punctuation"]["per_label"]["TITLE"],
        "pred_punct_macro": (
            report["capitalization_on_predicted_punctuation"]["capitalization_macro_f1"]
            if report["capitalization_on_predicted_punctuation"]
            else None
        ),
        "pred_punct_TITLE": (
            report["capitalization_on_predicted_punctuation"]["per_label"]["TITLE"]
            if report["capitalization_on_predicted_punctuation"]
            else None
        ),
        "preserve_fail": preserve_fail,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
