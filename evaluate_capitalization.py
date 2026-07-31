#!/usr/bin/env python3
"""Evaluate capitalization model on processed JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset
from transformers import DataCollatorForTokenClassification, Trainer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_capitalization.constants import IGNORE_LABEL, LABEL2ID  # noqa: E402
from kurmanji_capitalization.dataset import load_processed_jsonl  # noqa: E402
from kurmanji_capitalization.label_alignment import align_labels_to_first_subtoken  # noqa: E402
from kurmanji_capitalization.metrics import compute_capitalization_metrics, flatten_predictions  # noqa: E402
from kurmanji_capitalization.model import load_model, load_tokenizer  # noqa: E402


def tokenize_batch(examples, tokenizer, max_length: int):
    enc = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    all_labels = []
    for i, labs in enumerate(examples["labels"]):
        word_ids = enc.word_ids(batch_index=i)
        label_ids = []
        for x in labs:
            if x == IGNORE_LABEL or x == -100:
                label_ids.append(-100)
            else:
                label_ids.append(LABEL2ID[x])
        all_labels.append(align_labels_to_first_subtoken(word_ids, label_ids))
    enc["labels"] = all_labels
    return enc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/capitalization-v1.yaml"))
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    max_length = int(cfg["model"]["max_length"])
    rows = load_processed_jsonl(args.data)
    tokenizer = load_tokenizer(str(args.model))
    model = load_model(str(args.model))

    ds = Dataset.from_list(rows)
    tok = ds.map(
        lambda b: tokenize_batch(b, tokenizer, max_length),
        batched=True,
        remove_columns=ds.column_names,
    )
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = Trainer(model=model, data_collator=collator, tokenizer=tokenizer)
    pred = trainer.predict(tok)
    y_true, y_pred = flatten_predictions(pred.predictions, pred.label_ids)
    metrics = compute_capitalization_metrics(y_true, y_pred)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
