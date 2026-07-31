#!/usr/bin/env python3
"""Restore Kurmanji capitalization (sentence rule + XLM-R names/acronyms)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_capitalization.inference import CapitalizationRestorer  # noqa: E402
from kurmanji_capitalization.sentence_rule import capitalize_sentence_starts  # noqa: E402
from kurmanji_capitalization.casing import kurmanji_lower  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/capitalization-v1.yaml"))
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--input-file", type=Path, default=None)
    p.add_argument("--output-file", type=Path, default=None)
    p.add_argument("--json", action="store_true", help="Emit per-token predictions")
    p.add_argument("--rule-only", action="store_true", help="Only sentence-start rule")
    args = p.parse_args()

    if args.text is None and args.input_file is None:
        p.error("Provide --text or --input-file")

    text = args.text if args.text is not None else args.input_file.read_text(encoding="utf-8")

    if args.rule_only:
        out = capitalize_sentence_starts(kurmanji_lower(text))
        if args.output_file:
            args.output_file.write_text(out, encoding="utf-8")
        else:
            print(out)
        return 0

    cfg = {}
    if args.config.exists():
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inf = cfg.get("inference", {})
    conf = inf.get("minimum_confidence", {})
    restorer = CapitalizationRestorer(
        str(args.model),
        max_length=int(inf.get("max_length", 256)),
        overlap_words=int(inf.get("overlap_words", 32)),
        batch_size=int(inf.get("batch_size", 16)),
        title_threshold=float(conf.get("TITLE", 0.80)),
        upper_threshold=float(conf.get("UPPER", 0.90)),
    )

    if args.json:
        payload = restorer.predict_tokens(text)
        out = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        out = restorer.restore(text)

    if args.output_file:
        args.output_file.write_text(out, encoding="utf-8")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
