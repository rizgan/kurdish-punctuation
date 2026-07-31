#!/usr/bin/env python3
"""Production text restoration pipeline for Kurmanji.

```text
raw ASR / lowercased text
→ punctuation model
→ sentence-start rule (inside capitalization)
→ capitalization model
```

Examples:

```powershell
python restore_text.py `
  --punctuation-model models/punctuation/kurmanji-xlm-r-base-v2 `
  --capitalization-model models/capitalization/kurmanji-xlm-r-base-v1 `
  --input input.txt `
  --output output.txt

python restore_text.py --full --text "ez li amedê dijîm navê min azad e"
python restore_text.py --punctuation-only --text "ez li amedê dijîm"
python restore_text.py --capitalization-only --text "ez li amedê dijîm. navê min azad e."
python restore_text.py --full --json-output --text "ez li amedê dijîm navê min azad e"
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_capitalization.casing import kurmanji_lower  # noqa: E402
from kurmanji_capitalization.inference import CapitalizationRestorer  # noqa: E402
from kurmanji_capitalization.sentence_rule import capitalize_sentence_starts  # noqa: E402
from kurmanji_punctuation.inference import PunctuationRestorer  # noqa: E402


DEFAULT_PUNCT = Path("models/punctuation/kurmanji-xlm-r-base-v2")
DEFAULT_CAP = Path("models/capitalization/kurmanji-xlm-r-base-v1")


def load_yaml(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_input(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.input is not None:
        return args.input.read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Provide --text, --input, or pipe stdin")


def build_punctuation(path: Path, cfg: dict, device: str | None) -> PunctuationRestorer:
    inf = cfg.get("inference", {})
    return PunctuationRestorer(
        model_path=str(path),
        device=device,
        max_length=int(inf.get("max_length", 256)),
        overlap_words=int(inf.get("overlap_words", 32)),
        batch_size=int(inf.get("batch_size", 16)),
        minimum_confidence=inf.get("minimum_confidence"),
    )


def build_capitalization(path: Path, cfg: dict, device: str | None) -> CapitalizationRestorer:
    inf = cfg.get("inference", {})
    conf = inf.get("minimum_confidence", {})
    return CapitalizationRestorer(
        model_path=str(path),
        device=device,
        max_length=int(inf.get("max_length", 256)),
        overlap_words=int(inf.get("overlap_words", 32)),
        batch_size=int(inf.get("batch_size", 16)),
        title_threshold=float(conf.get("TITLE", 0.80)),
        upper_threshold=float(conf.get("UPPER", 0.90)),
        minimum_confidence=conf,
    )


def run_pipeline(
    text: str,
    *,
    mode: str,
    punct: PunctuationRestorer | None,
    cap: CapitalizationRestorer | None,
    json_output: bool,
) -> str | dict[str, Any]:
    stages: dict[str, Any] = {"input": text}

    if mode == "capitalization-only":
        assert cap is not None
        after_rule = capitalize_sentence_starts(kurmanji_lower(text))
        stages["after_sentence_rule"] = after_rule
        if json_output:
            tokens = cap.predict_tokens(text)
            stages["capitalization_tokens"] = tokens
            stages["output"] = cap.restore(text)
            return stages
        return cap.restore(text)

    assert punct is not None
    punctuated = punct.restore(text)
    stages["after_punctuation"] = punctuated

    if mode == "punctuation-only":
        if json_output:
            stages["punctuation_tokens"] = punct.predict_tokens(text)
            stages["output"] = punctuated
            return stages
        return punctuated

    # full
    assert cap is not None
    after_rule = capitalize_sentence_starts(kurmanji_lower(punctuated))
    stages["after_sentence_rule"] = after_rule
    final = cap.restore(punctuated)
    stages["output"] = final
    if json_output:
        stages["punctuation_tokens"] = punct.predict_tokens(text)
        stages["capitalization_tokens"] = cap.predict_tokens(punctuated)
        # Compact change list with confidence
        changes = []
        for row in stages["punctuation_tokens"]:
            if row.get("punctuation"):
                changes.append(
                    {
                        "stage": "punctuation",
                        "token": row["token"],
                        "label": row["label"],
                        "value": row["punctuation"],
                        "confidence": row["confidence"],
                    }
                )
        for row in stages["capitalization_tokens"]:
            if row["predicted_label"] != "KEEP" and not row.get("protected"):
                changes.append(
                    {
                        "stage": "capitalization",
                        "token_before": row["token_after_rule"],
                        "token_after": row["token_after"],
                        "label": row["predicted_label"],
                        "confidence": row["confidence"],
                        "sentence_start": row.get("sentence_start", False),
                    }
                )
        stages["changes"] = changes
        return stages
    return final


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", default=False, help="Punctuation + capitalization (default)")
    mode.add_argument("--punctuation-only", action="store_true")
    mode.add_argument("--capitalization-only", action="store_true")

    p.add_argument("--punctuation-model", type=Path, default=DEFAULT_PUNCT)
    p.add_argument("--capitalization-model", type=Path, default=DEFAULT_CAP)
    p.add_argument("--punctuation-config", type=Path, default=Path("config.yaml"))
    p.add_argument("--capitalization-config", type=Path, default=Path("configs/capitalization-v1.yaml"))
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--json-output", action="store_true", help="Emit stages + per-change confidence JSON")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    if args.punctuation_only:
        run_mode = "punctuation-only"
    elif args.capitalization_only:
        run_mode = "capitalization-only"
    else:
        run_mode = "full"

    text = read_input(args)

    punct = None
    cap = None
    if run_mode in ("full", "punctuation-only"):
        if not args.punctuation_model.exists():
            raise SystemExit(f"Punctuation model not found: {args.punctuation_model}")
        punct = build_punctuation(
            args.punctuation_model, load_yaml(args.punctuation_config), args.device
        )
    if run_mode in ("full", "capitalization-only"):
        if not args.capitalization_model.exists():
            raise SystemExit(f"Capitalization model not found: {args.capitalization_model}")
        cap = build_capitalization(
            args.capitalization_model, load_yaml(args.capitalization_config), args.device
        )

    result = run_pipeline(
        text,
        mode=run_mode,
        punct=punct,
        cap=cap,
        json_output=args.json_output,
    )

    if args.json_output:
        out = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        out = str(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out if out.endswith("\n") else out + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
