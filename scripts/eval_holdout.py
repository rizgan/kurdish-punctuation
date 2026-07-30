#!/usr/bin/env python3
"""Hold-out evaluation + qualitative ASR-like samples."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kurdish_punctuation.labels import PUNCT_LABELS  # noqa: E402
from kurdish_punctuation.restore import KurdishPunctuationModel  # noqa: E402
from kurdish_punctuation.text_utils import strip_punctuation_for_asr_like  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def labels_from_restore(model: KurdishPunctuationModel, words: list[str]) -> list[str]:
    text = " ".join(words)
    labeled = model.predict(text)
    # Align by position when lengths match; otherwise pad/truncate.
    preds = [lab for _w, lab, _s in labeled]
    if len(preds) < len(words):
        preds += ["0"] * (len(words) - len(preds))
    return preds[: len(words)]


def f1_report(y_true: list[str], y_pred: list[str]) -> dict:
    out = {}
    for lab in PUNCT_LABELS:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b != lab)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a != lab and b == lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(1 for x in y_true if x == lab)}
    punct = [out[x]["f1"] for x in PUNCT_LABELS if x != "0"]
    out["macro_punct_f1"] = sum(punct) / len(punct) if punct else 0.0
    out["accuracy"] = sum(1 for a, b in zip(y_true, y_pred) if a == b) / max(1, len(y_true))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="checkpoints/kmr-fullstop")
    p.add_argument("--test-file", type=Path, default=Path("data/processed/test.jsonl"))
    p.add_argument("--examples", type=Path, default=Path("examples/asr_samples.txt"))
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--out", type=Path, default=Path("outputs/eval_report.json"))
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    model = KurdishPunctuationModel(model=args.model, device=(-1 if args.cpu else None))

    report: dict = {"model": args.model, "domain_note": "Wiki hold-out ≠ TV/YouTube ASR domain."}

    if args.test_file.exists():
        rows = load_jsonl(args.test_file)[: args.max_samples]
        y_true: list[str] = []
        y_pred: list[str] = []
        for row in rows:
            words, labels = row["words"], row["labels"]
            preds = labels_from_restore(model, words)
            y_true.extend(labels)
            y_pred.extend(preds)
        report["holdout"] = f1_report(y_true, y_pred)
        report["holdout"]["n_sentences"] = len(rows)
        report["holdout"]["label_support"] = dict(Counter(y_true))
        print("Hold-out macro punct F1:", round(report["holdout"]["macro_punct_f1"], 4))
    else:
        print(f"No test file at {args.test_file} — skipping hold-out.")

    qualitative = []
    if args.examples.exists():
        for line in args.examples.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Allow "gold | asr-like" or just asr-like / punctuated gold.
            if "|" in line:
                gold, asr = [x.strip() for x in line.split("|", 1)]
            else:
                gold = line
                asr = strip_punctuation_for_asr_like(line)
            pred = model.restore_punctuation(asr)
            qualitative.append({"input": asr, "prediction": pred, "reference": gold})
            print("---")
            print("IN :", asr)
            print("OUT:", pred)
            if gold != asr:
                print("REF:", gold)
    report["qualitative"] = qualitative

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
