#!/usr/bin/env python3
"""Evaluate a trained checkpoint on processed JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
from datasets import Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_punctuation.constants import ID2LABEL, LABEL2ID, LABELS  # noqa: E402
from kurmanji_punctuation.dataset import load_processed_jsonl  # noqa: E402
from kurmanji_punctuation.label_alignment import align_labels_to_last_subtoken  # noqa: E402
from kurmanji_punctuation.metrics import (  # noqa: E402
    compute_punctuation_metrics,
    format_classification_report,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    max_length = int(cfg["model"]["max_length"])
    batch_size = int(cfg["training"]["eval_batch_size"])
    out_dir = args.output_dir or Path(cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_processed_jsonl(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    def _tok(batch):
        enc = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )
        aligned = []
        for i, labs in enumerate(batch["labels"]):
            word_ids = enc.word_ids(batch_index=i)
            label_ids = [LABEL2ID[x] for x in labs]
            aligned.append(align_labels_to_last_subtoken(word_ids, label_ids))
        enc["labels"] = aligned
        return enc

    ds = Dataset.from_list(rows).map(_tok, batched=True, remove_columns=["id", "article_id", "tokens", "labels"])
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collator)

    y_true: list[str] = []
    y_pred: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.cpu().numpy()
            label_np = labels.numpy()
            pred_ids = np.argmax(logits, axis=-1)
            for pred_row, lab_row in zip(pred_ids, label_np):
                for p_i, l_i in zip(pred_row, lab_row):
                    if int(l_i) == -100:
                        continue
                    y_true.append(ID2LABEL[int(l_i)])
                    y_pred.append(ID2LABEL[int(p_i)])

    metrics = compute_punctuation_metrics(y_true, y_pred)
    report = format_classification_report(metrics)

    (out_dir / "evaluation_report.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    with (out_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([""] + LABELS)
        for i, lab in enumerate(LABELS):
            w.writerow([lab] + list(metrics["confusion_matrix"][i]))

    print(report)
    print(f"punctuation_macro_f1={metrics['punctuation_macro_f1']:.4f}")
    print(f"Wrote reports under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
