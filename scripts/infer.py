#!/usr/bin/env python3
"""Restore punctuation for a string, a text file, or omnilingual ASR JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def words_from_omnilingual_json(path: Path) -> str:
    """
    Read omnilingual / best_gpu style JSON (*.wav___.json) and join `words`.
    Accepts either a list of word dicts or a top-level {"words": [...]} object.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "words" in data:
            words = data["words"]
        elif "text" in data:
            return str(data["text"])
        else:
            raise KeyError(f"No 'words' or 'text' in {path}")
    else:
        words = data

    out = []
    for w in words:
        if isinstance(w, str):
            out.append(w)
        elif isinstance(w, dict):
            out.append(w.get("word") or w.get("text") or w.get("token") or "")
        else:
            out.append(str(w))
    return " ".join(x for x in out if x).strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="checkpoints/kmr-fullstop", help="HF id or local path")
    p.add_argument("--text", type=str, default=None, help="Input string")
    p.add_argument("--file", type=Path, default=None, help="UTF-8 text file (one block)")
    p.add_argument(
        "--asr-json",
        type=Path,
        default=None,
        help="omnilingual ASR JSON (words from *.wav___.json)",
    )
    p.add_argument("--out", type=Path, default=None, help="Optional output file")
    p.add_argument(
        "--backend",
        choices=("local", "package"),
        default="local",
        help="local=KurdishPunctuationModel; package=deepmultilingualpunctuation.PunctuationModel",
    )
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if args.text:
        raw = args.text
    elif args.file:
        raw = args.file.read_text(encoding="utf-8").strip()
    elif args.asr_json:
        raw = words_from_omnilingual_json(args.asr_json)
    else:
        print("Provide --text, --file, or --asr-json", file=sys.stderr)
        return 2

    if args.backend == "package":
        from deepmultilingualpunctuation import PunctuationModel

        device_kw = {}
        model = PunctuationModel(model=args.model)
        restored = model.restore_punctuation(raw)
    else:
        from kurdish_punctuation.restore import KurdishPunctuationModel

        device = -1 if args.cpu else None
        model = KurdishPunctuationModel(model=args.model, device=device)
        restored = model.restore_punctuation(raw)

    print(restored)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(restored + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
