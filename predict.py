#!/usr/bin/env python3
"""Restore punctuation for a string or a text file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_punctuation.inference import PunctuationRestorer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--input-file", type=Path, default=None)
    p.add_argument("--output-file", type=Path, default=None)
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--show-tokens", action="store_true")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    cfg = {}
    if args.config.exists():
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    inf = cfg.get("inference", {})

    if args.text is not None:
        raw = args.text
    elif args.input_file is not None:
        raw = args.input_file.read_text(encoding="utf-8")
    else:
        print("Provide --text or --input-file", file=sys.stderr)
        return 2

    restorer = PunctuationRestorer(
        model_path=args.model,
        device=args.device,
        max_length=int(inf.get("max_length", 256)),
        overlap_words=int(inf.get("overlap_words", 32)),
        batch_size=int(inf.get("batch_size", 16)),
        minimum_confidence=inf.get("minimum_confidence"),
    )

    if args.show_tokens:
        tokens = restorer.predict_tokens(raw)
        print(json.dumps(tokens, ensure_ascii=False, indent=2))
        restored = restorer.restore(raw)
    else:
        restored = restorer.restore(raw)

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(restored + "\n", encoding="utf-8")
    print(restored)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
