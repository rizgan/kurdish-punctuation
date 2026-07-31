#!/usr/bin/env python3
"""
Tune per-class confidence thresholds on validation (maximize F1 per label).

Sweeps 0.30..0.90 and writes recommended thresholds into a JSON report.
Does not overwrite config.yaml unless --write-config is passed.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_punctuation.constants import LABELS  # noqa: E402
from kurmanji_punctuation.dataset import load_processed_jsonl  # noqa: E402
from kurmanji_punctuation.inference import PunctuationRestorer  # noqa: E402
from kurmanji_punctuation.metrics import compute_punctuation_metrics  # noqa: E402


def _f1_for_label(y_true: list[str], y_pred: list[str], lab: str) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("outputs/punctuation-xlm-r-base/best"))
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--data", type=Path, default=Path("data/processed/validation.jsonl"))
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--output", type=Path, default=Path("outputs/punctuation-xlm-r-base/threshold_tuning.json"))
    p.add_argument("--write-config", action="store_true")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inf = cfg.get("inference", {})
    rows = load_processed_jsonl(args.data)[: args.max_samples]

    restorer = PunctuationRestorer(
        model_path=str(args.model),
        device=args.device,
        max_length=int(inf.get("max_length", 256)),
        overlap_words=int(inf.get("overlap_words", 32)),
        minimum_confidence={k: 0.0 for k in ("COMMA", "PERIOD", "QUESTION", "EXCLAMATION")},
    )

    # Collect gold labels and probability rows (concatenated samples; no EOS leakage concern
    # for threshold sweep on held-out val — we still use sample format, but optimize F1).
    all_true: list[str] = []
    all_probs: list[np.ndarray] = []
    print(f"Scoring {len(rows)} validation samples …")
    for i, row in enumerate(rows):
        tokens = row["tokens"]
        labels = row["labels"]
        logits = restorer.mean_logits_for_words(tokens)
        probs = restorer.softmax_rows(logits)
        if len(probs) != len(labels):
            continue
        all_true.extend(labels)
        all_probs.extend(list(probs))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rows)}")

    probs_mat = np.stack(all_probs, axis=0)
    grid = [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]
    punct = [lab for lab in LABELS if lab != "O"]

    # Greedy: independently pick best threshold per class (argmax among punct + O with floor).
    best_thr: dict[str, float] = {}
    sweeps: dict[str, list[dict]] = {}
    for lab in punct:
        sweeps[lab] = []
        best_f1 = -1.0
        best_t = 0.55
        for t in grid:
            thr_map = {x: 1.0 for x in punct}  # block other punct temporarily? No —
            # Decode: take argmax; if punct conf < its thr → O.
            # For independent sweep of `lab`, vary only lab's thr; others use current best or 0.55.
            trial = {x: best_thr.get(x, 0.55) for x in punct}
            trial[lab] = t
            preds = []
            for row in probs_mat:
                pred_id = int(row.argmax())
                pred = LABELS[pred_id]
                conf = float(row[pred_id])
                if pred != "O" and conf < float(trial.get(pred, 0.0)):
                    pred = "O"
                preds.append(pred)
            f1 = _f1_for_label(all_true, preds, lab)
            sweeps[lab].append({"threshold": t, "f1": f1})
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thr[lab] = best_t
        print(f"{lab}: best_thr={best_t:.2f} f1={best_f1:.4f}")

    # Final metrics with tuned thresholds
    preds_final = []
    for row in probs_mat:
        pred_id = int(row.argmax())
        pred = LABELS[pred_id]
        conf = float(row[pred_id])
        if pred != "O" and conf < float(best_thr.get(pred, 0.0)):
            pred = "O"
        preds_final.append(pred)
    metrics = compute_punctuation_metrics(all_true, preds_final)

    # Baseline with config thresholds
    base_thr = inf.get("minimum_confidence") or {}
    preds_base = []
    for row in probs_mat:
        pred_id = int(row.argmax())
        pred = LABELS[pred_id]
        conf = float(row[pred_id])
        if pred != "O" and conf < float(base_thr.get(pred, 0.0)):
            pred = "O"
        preds_base.append(pred)
    metrics_base = compute_punctuation_metrics(all_true, preds_base)

    report = {
        "n_tokens": len(all_true),
        "n_samples": len(rows),
        "grid": grid,
        "baseline_thresholds": base_thr,
        "baseline_macro_f1": metrics_base["punctuation_macro_f1"],
        "tuned_thresholds": best_thr,
        "tuned_macro_f1": metrics["punctuation_macro_f1"],
        "tuned_per_label": metrics["per_label"],
        "sweeps": sweeps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("baseline_macro_f1", "tuned_macro_f1", "tuned_thresholds")}, indent=2))
    print(f"Wrote {args.output}")

    if args.write_config:
        new_cfg = copy.deepcopy(cfg)
        new_cfg.setdefault("inference", {})["minimum_confidence"] = best_thr
        args.config.write_text(yaml.safe_dump(new_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"Updated thresholds in {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
