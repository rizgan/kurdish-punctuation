#!/usr/bin/env python3
"""
Smoke-test for frozen kurmanji-punctuation-xlm-r-base-v1.0.

Loads the frozen model path, runs fixed examples, asserts text preservation
and that each restore() succeeds without crashing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_punctuation.inference import PunctuationRestorer, TextPreservationError  # noqa: E402
from kurmanji_punctuation.text_utils import validate_text_preservation  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "punctuation" / "kurmanji-xlm-r-base-v1"

# Fixed regression probes (ASR-like: no punctuation, mixed case preserved by design = lowercase ASR).
SMOKE_CASES = [
    {
        "id": "comma_clause",
        "input": "ez îro çûm bajarê lê baran dibariya",
        "must_contain_any": [",", "."],
    },
    {
        "id": "period_end",
        "input": "li kurdistanê gelek çiyayên bilind hene",
        "must_contain_any": ["."],
    },
    {
        "id": "multi_sentence_words",
        "input": "ez li amedê dijîm navê min azad e ew li hewlêrê dixebite",
        "must_contain_any": ["."],
    },
    {
        "id": "questionish",
        "input": "tu kengî hatî",
        # Preservation only — QUESTION is a known weak class in v1; no hard punct assert.
        "must_contain_any": ["tu", "kengî", "hatî"],
    },
    {
        "id": "kurmanji_diacritics",
        "input": "çêşîû û çiyayên bilind li wir hene",
        "must_contain_any": ["ç", "ê", "î", "ş", "û"],
    },
    {
        "id": "url_number_email",
        "input": "binêre https://ku.wikipedia.org û name@example.com û 42",
        "must_contain_any": ["https://ku.wikipedia.org", "name@example.com", "42"],
    },
    {
        "id": "empty",
        "input": "",
        "must_contain_any": [],
    },
    {
        "id": "single_token",
        "input": "were",
        "must_contain_any": ["were"],
    },
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    if not args.model.exists():
        print(f"FAIL: model path missing: {args.model}", file=sys.stderr)
        return 2

    restorer = PunctuationRestorer(model_path=str(args.model), device=args.device)
    results = []
    failed = 0

    for case in SMOKE_CASES:
        inp = case["input"]
        row = {"id": case["id"], "input": inp, "ok": False}
        try:
            out = restorer.restore(inp)
            row["output"] = out
            preserved = validate_text_preservation(inp, out) if inp else (out == "")
            row["text_preservation"] = preserved
            if not preserved:
                raise TextPreservationError(f"preservation failed for {case['id']}")
            for needle in case.get("must_contain_any") or []:
                if needle and needle not in out and needle not in inp:
                    # For punctuation needles, only require presence in output.
                    if needle in {".", ",", "?", "!"}:
                        if needle not in out:
                            raise AssertionError(f"expected one of punctuation including {needle!r}, got {out!r}")
                    else:
                        raise AssertionError(f"expected {needle!r} preserved in {out!r}")
            # If must_contain_any is a list of punct options, at least one should appear (non-empty inputs with punct expectation)
            punct_needles = [x for x in (case.get("must_contain_any") or []) if x in {".", ",", "?", "!"}]
            if punct_needles and inp and not any(x in out for x in punct_needles):
                raise AssertionError(f"expected one of {punct_needles} in {out!r}")
            row["ok"] = True
            print(f"PASS {case['id']}: {out!r}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            row["ok"] = False
            row["error"] = str(exc)
            print(f"FAIL {case['id']}: {exc}", file=sys.stderr)
        results.append(row)

    report = {
        "model": str(args.model),
        "n_cases": len(SMOKE_CASES),
        "n_failed": failed,
        "all_passed": failed == 0,
        "cases": results,
    }
    out_path = args.json_out or (args.model / "smoke_test_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
