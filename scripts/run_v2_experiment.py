#!/usr/bin/env python3
"""Orchestrate a v2 experiment with frozen-hash gates.

Usage:
  python scripts/run_v2_experiment.py --experiment v2-exp-01 --config configs/v2-exp-01.yaml

Stops on any pre-flight failure (weak-model safety rule).
Does not auto-edit docs unless --update-docs is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "configs" / "v2-base-frozen.yaml"
ALLOWED_DIFF_PATHS = {
    ("experiment", "id"),
    ("experiment", "output_dir"),
    ("experiment", "model_id"),
    ("project", "output_dir"),
    ("data", "additional_question_corpus"),
    ("data", "sampling_weight"),
    ("dataset_hash",),
    ("extends",),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def flatten(d: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    out: dict[tuple[str, ...], Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, prefix + (str(k),)))
    else:
        out[prefix] = d
    return out


def merge_extends(cfg_path: Path) -> dict[str, Any]:
    cfg = load_yaml(cfg_path)
    extends = cfg.pop("extends", None)
    if extends:
        base = load_yaml(ROOT / extends)
        # shallow+one-level merge for known sections
        merged = dict(base)
        for k, v in cfg.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        return merged
    return cfg


def assert_config_allowed(exp_cfg: dict[str, Any], frozen: dict[str, Any]) -> None:
    flat_exp = flatten(exp_cfg)
    flat_fr = flatten(frozen)
    bad: list[str] = []
    for key, val in flat_exp.items():
        if key in ALLOWED_DIFF_PATHS or key[:1] == ("extends",):
            continue
        if key in flat_fr and flat_fr[key] != val:
            bad.append(".".join(key))
        elif key not in flat_fr and key not in ALLOWED_DIFF_PATHS:
            # new keys only allowed under experiment/data sampling
            if key[0] not in {"experiment", "data"} and key != ("dataset_hash",):
                bad.append(".".join(key) + " (unexpected)")
    # Check frozen keys that must match when present in both
    for key, val in flat_fr.items():
        if key in ALLOWED_DIFF_PATHS:
            continue
        if key in flat_exp and flat_exp[key] != val:
            bad.append(".".join(key))
    if bad:
        raise SystemExit(f"[gate] FAIL: disallowed config diffs: {sorted(set(bad))}")


def require_file(path: Path, label: str) -> str:
    if not path.exists():
        raise SystemExit(f"[gate] FAIL: missing {label}: {path}")
    return sha256_file(path)


def preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    frozen_model = ROOT / "models" / "punctuation" / "kurmanji-xlm-r-base-v1"
    run_info = frozen_model / "run_info.json"
    require_file(frozen_model / "config.json", "v1 config")
    require_file(run_info, "v1 run_info")
    v1 = json.loads(run_info.read_text(encoding="utf-8"))

    val = ROOT / "data" / "processed" / "validation.jsonl"
    test = ROOT / "data" / "processed" / "test.jsonl"
    val_h = require_file(val, "validation.jsonl")
    test_h = require_file(test, "test.jsonl")

    expected = {x.split(":", 1)[0]: x.split(":", 1)[1] for x in v1.get("dataset_files_sha256", []) if ":" in x}
    if expected.get("validation.jsonl") and expected["validation.jsonl"] != val_h:
        raise SystemExit("[gate] FAIL: validation.jsonl hash != v1 frozen hash")
    if expected.get("test.jsonl") and expected["test.jsonl"] != test_h:
        raise SystemExit("[gate] FAIL: test.jsonl hash != v1 frozen hash")

    leak = ROOT / "data" / "v2_question_processed" / "leakage_report.json"
    require_file(leak, "leakage_report.json")
    leak_rep = json.loads(leak.read_text(encoding="utf-8"))
    if leak_rep.get("test_contamination") is True:
        raise SystemExit("[gate] FAIL: test_contamination=true")

    qcorp = cfg.get("data", {}).get("additional_question_corpus")
    if qcorp:
        require_file(ROOT / qcorp, "additional_question_corpus")

    info = {
        "v1_dataset_hash": v1.get("dataset_hash"),
        "validation_sha256": val_h,
        "test_sha256": test_h,
        "leakage_test_contamination": leak_rep.get("test_contamination"),
        "sampling_weight": cfg.get("data", {}).get("sampling_weight"),
        "experiment_id": (cfg.get("experiment") or {}).get("id"),
    }
    print(json.dumps({"preflight": info}, ensure_ascii=False, indent=2))
    return info


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, choices=["v2-exp-01", "v2-exp-02", "v2-exp-02a", "v2-exp-02b"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--skip-train", action="store_true", help="Preflight only")
    p.add_argument("--update-docs", action="store_true")
    args = p.parse_args()

    frozen = load_yaml(FROZEN)
    cfg = merge_extends(args.config if args.config.is_absolute() else ROOT / args.config)
    assert_config_allowed(cfg, frozen)
    preflight(cfg)

    if args.skip_train:
        print("[ok] preflight passed; train skipped")
        return 0

    out_dir = Path(cfg.get("project", {}).get("output_dir") or cfg.get("experiment", {}).get("output_dir"))
    print(
        "[todo] Training orchestration hooks into scripts/train.py + evaluate_long_text.py.\n"
        f"       Configure output_dir={out_dir} after Stage 6 dataset build is complete.\n"
        "       Refusing to start train until processed_v2_question dataset gate passes."
    )
    processed_v2 = ROOT / "data" / "processed_v2_question"
    stats = processed_v2 / "statistics.json"
    if not stats.exists():
        raise SystemExit("[gate] FAIL: data/processed_v2_question/statistics.json missing — finish Stage 6 first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
