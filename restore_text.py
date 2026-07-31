#!/usr/bin/env python3
"""Production Kurmanji text restoration CLI.

```text
input text
→ punctuation v2
→ sentence-start capitalization rule
→ capitalization v1
→ preservation checks
→ final text
```

Examples:

```powershell
python restore_text.py `
  --punctuation-model models/punctuation/kurmanji-xlm-r-base-v2 `
  --capitalization-model models/capitalization/kurmanji-xlm-r-base-v1 `
  --input input.txt `
  --output output.txt

python restore_text.py --mode full --text "ez li amedê dijîm navê min azad e"
python restore_text.py --mode punctuation --text "ez li amedê dijîm"
python restore_text.py --mode capitalization --text "ez li amedê dijîm. navê min azad e."
python restore_text.py --mode full --json-output --text "ez li amedê dijîm navê min azad e"
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

from kurmanji_capitalization.inference import CapitalizationRestorer  # noqa: E402
from kurmanji_pipeline.pipeline import (  # noqa: E402
    DEFAULT_TITLE_THRESHOLD,
    DEFAULT_UPPER_THRESHOLD,
    TextRestorationPipeline,
)
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


def build_capitalization(
    path: Path,
    cfg: dict,
    device: str | None,
    *,
    title_threshold: float,
    upper_threshold: float,
) -> CapitalizationRestorer:
    inf = cfg.get("inference", {})
    conf = dict(inf.get("minimum_confidence") or {})
    conf["TITLE"] = title_threshold
    conf["UPPER"] = upper_threshold
    return CapitalizationRestorer(
        model_path=str(path),
        device=device,
        max_length=int(inf.get("max_length", 256)),
        overlap_words=int(inf.get("overlap_words", 32)),
        batch_size=int(inf.get("batch_size", 16)),
        title_threshold=title_threshold,
        upper_threshold=upper_threshold,
        minimum_confidence=conf,
    )


def public_json(result_dict: dict[str, Any]) -> dict[str, Any]:
    """JSON contract for --json-output (stages + preservation)."""
    return {
        "input": result_dict["input"],
        "punctuated": result_dict.get("punctuated"),
        "sentence_capitalized": result_dict.get("sentence_capitalized"),
        "output": result_dict["output"],
        "preservation": result_dict["preservation"],
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=("full", "punctuation", "capitalization"),
        default="full",
        help="Pipeline mode (default: full)",
    )
    p.add_argument("--punctuation-model", type=Path, default=DEFAULT_PUNCT)
    p.add_argument("--capitalization-model", type=Path, default=DEFAULT_CAP)
    p.add_argument("--punctuation-config", type=Path, default=Path("config.yaml"))
    p.add_argument(
        "--capitalization-config",
        type=Path,
        default=Path("configs/capitalization-v1.yaml"),
    )
    p.add_argument(
        "--title-threshold",
        type=float,
        default=DEFAULT_TITLE_THRESHOLD,
        help=f"TITLE confidence floor (default: {DEFAULT_TITLE_THRESHOLD})",
    )
    p.add_argument(
        "--upper-threshold",
        type=float,
        default=DEFAULT_UPPER_THRESHOLD,
        help=f"UPPER confidence floor (default: {DEFAULT_UPPER_THRESHOLD})",
    )
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--json-output",
        action="store_true",
        help="Emit stages + preservation diagnostics as JSON",
    )
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    text = read_input(args)
    mode = args.mode

    punct = None
    cap = None
    if mode in ("full", "punctuation"):
        if not args.punctuation_model.exists():
            raise SystemExit(f"Punctuation model not found: {args.punctuation_model}")
        punct = build_punctuation(
            args.punctuation_model, load_yaml(args.punctuation_config), args.device
        )
    if mode in ("full", "capitalization"):
        if not args.capitalization_model.exists():
            raise SystemExit(f"Capitalization model not found: {args.capitalization_model}")
        cap = build_capitalization(
            args.capitalization_model,
            load_yaml(args.capitalization_config),
            args.device,
            title_threshold=args.title_threshold,
            upper_threshold=args.upper_threshold,
        )

    pipeline = TextRestorationPipeline(punctuation=punct, capitalization=cap)
    result = pipeline.run(text, mode=mode)
    payload = result.to_dict()

    if args.json_output:
        out = json.dumps(public_json(payload), ensure_ascii=False, indent=2)
    else:
        out = result.output

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out if out.endswith("\n") else out + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
