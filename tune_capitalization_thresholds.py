#!/usr/bin/env python3
"""Tune TITLE/UPPER confidence thresholds on capitalization validation windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_capitalization.constants import IGNORE_LABEL, LABEL2ID, LABELS  # noqa: E402
from kurmanji_capitalization.dataset import load_processed_jsonl  # noqa: E402
from kurmanji_capitalization.inference import CapitalizationRestorer  # noqa: E402
from kurmanji_capitalization.metrics import compute_capitalization_metrics  # noqa: E402


def decode_batch(
    probs_list: list[np.ndarray],
    sentence_starts: list[bool],
    protected: list[bool],
    title_thr: float,
    upper_thr: float,
) -> list[str]:
    out = []
    for probs, ss, prot in zip(probs_list, sentence_starts, protected):
        if prot:
            out.append("KEEP")
            continue
        pred_id = int(np.argmax(probs))
        label = LABELS[pred_id]
        conf = float(probs[pred_id])
        if ss:
            out.append("UPPER" if label == "UPPER" and conf >= upper_thr else "KEEP")
            continue
        if label == "TITLE" and conf >= title_thr:
            out.append("TITLE")
        elif label == "UPPER" and conf >= upper_thr:
            out.append("UPPER")
        else:
            out.append("KEEP")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("outputs/capitalization-xlm-r-base-v1/best"))
    p.add_argument("--config", type=Path, default=Path("configs/capitalization-v1.yaml"))
    p.add_argument("--data", type=Path, default=Path("data/processed_capitalization/validation.jsonl"))
    p.add_argument("--max-samples", type=int, default=2384)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/capitalization-xlm-r-base-v1/threshold_tuning.json"),
    )
    p.add_argument("--write-config", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inf = cfg.get("inference", {})
    rows = load_processed_jsonl(args.data)[: args.max_samples]

    # Collect raw probs with thresholds disabled (decode offline).
    restorer = CapitalizationRestorer(
        str(args.model),
        max_length=int(inf.get("max_length", 256)),
        overlap_words=int(inf.get("overlap_words", 32)),
        title_threshold=0.0,
        upper_threshold=0.0,
    )

    golds: list[str] = []
    probs_list: list[np.ndarray] = []
    sentence_starts: list[bool] = []
    protected: list[bool] = []

    print(f"Scoring {len(rows)} validation windows …", flush=True)
    for i, row in enumerate(rows):
        scored = restorer.score_tokens(row["tokens"])
        for gold, pred in zip(row["labels"], scored):
            if gold == IGNORE_LABEL:
                continue
            golds.append(gold)
            probs_list.append(
                np.array([pred["probs"][lab] for lab in LABELS], dtype=np.float32)
            )
            sentence_starts.append(bool(pred["sentence_start"]))
            protected.append(bool(pred["protected"]))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    title_grid = [round(x, 2) for x in np.arange(0.80, 0.97, 0.02)]
    upper_grid = [round(x, 2) for x in np.arange(0.85, 0.99, 0.02)]

    sweeps = []
    best = None
    # Prefer precision-oriented TITLE while keeping recall decent.
    for t in title_grid:
        for u in upper_grid:
            preds = decode_batch(probs_list, sentence_starts, protected, t, u)
            m = compute_capitalization_metrics(golds, preds)
            title = m["per_label"]["TITLE"]
            upper = m["per_label"]["UPPER"]
            keep = m["per_label"]["KEEP"]
            row = {
                "TITLE": t,
                "UPPER": u,
                "macro_f1": m["capitalization_macro_f1"],
                "TITLE_f1": title["f1"],
                "TITLE_precision": title["precision"],
                "TITLE_recall": title["recall"],
                "UPPER_f1": upper["f1"],
                "KEEP_f1": keep["f1"],
                "user_target": title["precision"] >= 0.90 and title["recall"] >= 0.85,
            }
            sweeps.append(row)
            # Rank: first user_target, then TITLE precision, then macro F1
            key = (
                1 if row["user_target"] else 0,
                row["TITLE_precision"],
                row["macro_f1"],
                row["TITLE_recall"],
            )
            if best is None or key > best[0]:
                best = (key, row)

    # Also track max-macro and max-TITLE-F1
    max_macro = max(sweeps, key=lambda r: r["macro_f1"])
    max_title_f1 = max(sweeps, key=lambda r: r["TITLE_f1"])
    precision_oriented = best[1] if best else max_macro

    report = {
        "n_tokens": len(golds),
        "n_windows": len(rows),
        "recommended_precision_oriented": precision_oriented,
        "max_macro_f1": max_macro,
        "max_TITLE_f1": max_title_f1,
        "default_080_090": next(
            (r for r in sweeps if r["TITLE"] == 0.80 and r["UPPER"] == 0.90), None
        ),
        "sweeps": sweeps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.write_config:
        cfg.setdefault("inference", {}).setdefault("minimum_confidence", {})
        cfg["inference"]["minimum_confidence"]["TITLE"] = precision_oriented["TITLE"]
        cfg["inference"]["minimum_confidence"]["UPPER"] = precision_oriented["UPPER"]
        args.config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"updated {args.config}")

    print(json.dumps({
        "recommended_precision_oriented": precision_oriented,
        "max_macro_f1": max_macro,
        "default_080_090": report["default_080_090"],
    }, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
