#!/usr/bin/env python3
"""Pre-release gates for the unified production pipeline.

Checks:
  * edge cases (empty, one word, questions, URLs, emails, numbers, hyphens, apostrophes)
  * already-punctuated input
  * near-idempotency: pipeline(pipeline(text)) preserves words + case-only vs pipeline(text)
  * optional: N Wikipedia articles (strip punct → restore → preservation)
  * optional: long-text concat (≥10k words)

```powershell
python scripts/smoke_test_production_pipeline.py
python scripts/smoke_test_production_pipeline.py --max-articles 400 --long-text-words 10000
```
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_capitalization.casing import kurmanji_lower  # noqa: E402
from kurmanji_capitalization.inference import CapitalizationRestorer  # noqa: E402
from kurmanji_capitalization.normalization import normalize_for_dataset  # noqa: E402
from kurmanji_capitalization.text_utils import validate_case_only_transformation  # noqa: E402
from kurmanji_pipeline.pipeline import (  # noqa: E402
    DEFAULT_TITLE_THRESHOLD,
    DEFAULT_UPPER_THRESHOLD,
    TextRestorationPipeline,
)
from kurmanji_punctuation.dataset import load_processed_jsonl, tokens_and_labels_from_text  # noqa: E402
from kurmanji_punctuation.inference import PunctuationRestorer  # noqa: E402
from kurmanji_punctuation.normalization import normalize_for_inference  # noqa: E402
from kurmanji_punctuation.text_utils import (  # noqa: E402
    extract_words,
    strip_supported_punctuation,
    validate_text_preservation,
)

EDGE_CASES: list[tuple[str, str]] = [
    ("empty", ""),
    ("one_word", "azad"),
    ("questions_chain", "tu baş î tu li ku dijî kengê em dê biçin"),
    ("url_email_number", "binêre https://ku.wikipedia.org û name@example.com û 42.5"),
    ("hyphen_apostrophe", "navê welat-parêz û dayê'ê"),
    ("already_punctuated", "Ez li Amedê dijîm. Navê min Azad e."),
    ("demo", "ez li amedê dijîm navê min azad e"),
]


def prepare_article_text(text: str) -> str:
    """Match long-text eval prep: normalize → word tokens → space-join → lower.

    Space-joining via the shared tokenizer avoids false preservation failures on
    agglutinations like ``21ê`` (digit+letter) that tokenize as two words.
    """
    text = normalize_for_dataset(text)
    text = text.replace("|", " ")
    text = " ".join(text.split())
    pair = tokens_and_labels_from_text(text)
    if not pair:
        return ""
    tokens, _labels = pair
    return kurmanji_lower(" ".join(tokens))


def near_idempotent(a: str, b: str) -> bool:
    """True if a second pass only moved punctuation / case, not word identity.

    New sentence boundaries can retitle words (``e`` → ``E``), so compare the
    Kurmanji-lowercased text after stripping supported punctuation.
    """
    if a == b:
        return True
    left = strip_supported_punctuation(kurmanji_lower(a))
    right = strip_supported_punctuation(kurmanji_lower(b))
    return left == right


def build_pipeline(
    punct_path: Path,
    cap_path: Path,
    *,
    title_threshold: float,
    upper_threshold: float,
    device: str | None,
) -> TextRestorationPipeline:
    return TextRestorationPipeline(
        punctuation=PunctuationRestorer(str(punct_path), device=device),
        capitalization=CapitalizationRestorer(
            str(cap_path),
            device=device,
            title_threshold=title_threshold,
            upper_threshold=upper_threshold,
            minimum_confidence={"TITLE": title_threshold, "UPPER": upper_threshold},
        ),
    )


def check_edge_cases(pipeline: TextRestorationPipeline) -> dict:
    rows = []
    failed = 0
    for name, text in EDGE_CASES:
        try:
            r1 = pipeline.run(text, mode="full")
            r2 = pipeline.run(r1.output, mode="full")
            exact = r2.output == r1.output
            near = near_idempotent(r1.output, r2.output)
            ok = (
                r1.preservation.get("punctuation_only", False)
                and r1.preservation.get("case_only", False)
                and near
            )
            # URL/email must keep host casing intact (no sentence-rule on internal dots).
            if name == "url_email_number":
                if "Wikipedia" in r1.output or ".Com" in r1.output:
                    ok = False
            if not ok:
                failed += 1
            rows.append(
                {
                    "name": name,
                    "ok": ok,
                    "idempotent_exact": exact,
                    "idempotent_near": near,
                    "preservation": r1.preservation,
                    "output": r1.output,
                    "second_pass": r2.output,
                }
            )
        except Exception as exc:  # noqa: BLE001 — collect per-case failures
            failed += 1
            rows.append({"name": name, "ok": False, "error": str(exc)})
    return {"failed": failed, "cases": rows}


def load_articles(raw: Path, keep: set[str] | None, limit: int, seed: int) -> list[dict]:
    articles: list[dict] = []
    with raw.open(encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if keep is not None and obj.get("article_id") not in keep:
                continue
            articles.append(obj)
            if keep is None and len(articles) >= max(limit * 5, limit):
                break
    rng = random.Random(seed)
    if len(articles) > limit:
        articles = rng.sample(articles, limit)
    return articles[:limit]


def check_articles(
    pipeline: TextRestorationPipeline,
    raw: Path,
    *,
    max_articles: int,
    seed: int,
    processed_split: Path | None,
) -> dict:
    keep = None
    if processed_split is not None and processed_split.exists():
        keep = {r["article_id"] for r in load_processed_jsonl(processed_split)}
    articles = load_articles(raw, keep, max_articles, seed)

    n_ok = 0
    n_fail = 0
    n_idempotent = 0
    n_near_idempotent = 0
    failures: list[dict] = []

    for art in articles:
        stripped = prepare_article_text(art.get("text") or "")
        if not extract_words(stripped):
            continue
        try:
            r1 = pipeline.run(stripped, mode="full")
            punct_ok = bool(r1.preservation["punctuation_only"])
            case_ok = bool(r1.preservation["case_only"])
            if r1.punctuated is not None:
                case_ok = case_ok and validate_case_only_transformation(
                    r1.punctuated, r1.output
                )
                punct_ok = punct_ok and validate_text_preservation(stripped, r1.punctuated)
            r2 = pipeline.run(r1.output, mode="full")
            exact = r2.output == r1.output
            near = near_idempotent(r1.output, r2.output)
            if exact:
                n_idempotent += 1
            if near:
                n_near_idempotent += 1
            if punct_ok and case_ok:
                n_ok += 1
            else:
                n_fail += 1
                if len(failures) < 20:
                    failures.append(
                        {
                            "article_id": art.get("article_id"),
                            "punctuation_only": punct_ok,
                            "case_only": case_ok,
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            if len(failures) < 20:
                failures.append({"article_id": art.get("article_id"), "error": str(exc)})

    total = n_ok + n_fail
    return {
        "total": total,
        "preservation_ok": n_ok,
        "preservation_fail": n_fail,
        "preservation_rate": (n_ok / total) if total else 1.0,
        "idempotent_exact": n_idempotent,
        "idempotent_near": n_near_idempotent,
        "idempotent_exact_rate": (n_idempotent / total) if total else 1.0,
        "idempotent_near_rate": (n_near_idempotent / total) if total else 1.0,
        "failures": failures,
    }


def check_long_text(
    pipeline: TextRestorationPipeline,
    raw: Path,
    *,
    min_words: int,
    seed: int,
    processed_split: Path | None,
) -> dict:
    keep = None
    if processed_split is not None and processed_split.exists():
        keep = {r["article_id"] for r in load_processed_jsonl(processed_split)}
    rng = random.Random(seed)
    chunks: list[str] = []
    words = 0
    with raw.open(encoding="utf-8") as fh:
        lines = fh.readlines()
    rng.shuffle(lines)
    for line in lines:
        art = json.loads(line)
        if keep is not None and art.get("article_id") not in keep:
            continue
        stripped = prepare_article_text(art.get("text") or "")
        w = extract_words(stripped)
        if not w:
            continue
        chunks.append(stripped)
        words += len(w)
        if words >= min_words:
            break

    blob = " ".join(chunks)
    blob = normalize_for_inference(blob)
    n_words = len(extract_words(blob))
    try:
        r1 = pipeline.run(blob, mode="full")
        r2 = pipeline.run(r1.output, mode="full")
        exact = r2.output == r1.output
        near = near_idempotent(r1.output, r2.output)
        return {
            "words": n_words,
            "chars": len(blob),
            "preservation": r1.preservation,
            "punctuation_only": r1.preservation["punctuation_only"],
            "case_only": r1.preservation["case_only"],
            "idempotent_exact": exact,
            "idempotent_near": near,
            "pass": (
                n_words >= min_words
                and r1.preservation["punctuation_only"]
                and r1.preservation["case_only"]
                and near
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "words": n_words,
            "chars": len(blob),
            "error": str(exc),
            "pass": False,
        }


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
        default=Path("models/capitalization/kurmanji-xlm-r-base-v1"),
    )
    p.add_argument("--raw", type=Path, default=Path("data/raw/wikipedia.jsonl"))
    p.add_argument(
        "--processed-split",
        type=Path,
        default=Path("data/processed/test.jsonl"),
        help="Prefer held-out test article_ids when present",
    )
    p.add_argument("--max-articles", type=int, default=0, help="0 = skip article sweep")
    p.add_argument("--long-text-words", type=int, default=0, help="0 = skip long-text")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--title-threshold", type=float, default=DEFAULT_TITLE_THRESHOLD)
    p.add_argument("--upper-threshold", type=float, default=DEFAULT_UPPER_THRESHOLD)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/production_pipeline_smoke.json"),
    )
    args = p.parse_args()

    pipeline = build_pipeline(
        args.punctuation_model,
        args.capitalization_model,
        title_threshold=args.title_threshold,
        upper_threshold=args.upper_threshold,
        device=args.device,
    )

    report: dict = {
        "title_threshold": args.title_threshold,
        "upper_threshold": args.upper_threshold,
        "punctuation_model": str(args.punctuation_model),
        "capitalization_model": str(args.capitalization_model),
    }

    print("=== edge cases ===", flush=True)
    edge = check_edge_cases(pipeline)
    report["edge_cases"] = edge
    print(json.dumps({"failed": edge["failed"], "n": len(edge["cases"])}, indent=2), flush=True)

    if args.max_articles > 0:
        print(f"=== articles ({args.max_articles}) ===", flush=True)
        art = check_articles(
            pipeline,
            args.raw,
            max_articles=args.max_articles,
            seed=args.seed,
            processed_split=args.processed_split,
        )
        report["articles"] = art
        print(
            json.dumps(
                {
                    "total": art["total"],
                    "preservation_rate": art["preservation_rate"],
                    "idempotent_exact_rate": art["idempotent_exact_rate"],
                    "idempotent_near_rate": art["idempotent_near_rate"],
                },
                indent=2,
            ),
            flush=True,
        )

    if args.long_text_words > 0:
        print(f"=== long text (≥{args.long_text_words} words) ===", flush=True)
        long = check_long_text(
            pipeline,
            args.raw,
            min_words=args.long_text_words,
            seed=args.seed,
            processed_split=args.processed_split,
        )
        report["long_text"] = long
        print(
            json.dumps(
                {
                    k: long[k]
                    for k in (
                        "words",
                        "pass",
                        "idempotent_exact",
                        "idempotent_near",
                        "preservation",
                        "error",
                    )
                    if k in long
                },
                indent=2,
            ),
            flush=True,
        )

    edge_pass = edge["failed"] == 0
    art_pass = True
    if "articles" in report:
        art_pass = (
            report["articles"]["preservation_fail"] == 0
            and report["articles"]["idempotent_near_rate"] >= 0.98
        )
    long_pass = True
    if "long_text" in report:
        long_pass = bool(report["long_text"]["pass"])

    report["pass"] = bool(edge_pass and art_pass and long_pass)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    print(f"PASS={report['pass']}", flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
