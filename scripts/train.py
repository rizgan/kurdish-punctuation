#!/usr/bin/env python3
"""Fine-tune a FullStop / token-classification checkpoint on Kurmanji Latin data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kurdish_punctuation.labels import DEFAULT_ID2LABEL, DEFAULT_LABEL2ID, PUNCT_LABELS  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_label_maps(base_model: str) -> tuple[dict[str, int], dict[int, str]]:
    try:
        cfg = AutoConfig.from_pretrained(base_model)
        if getattr(cfg, "label2id", None) and cfg.label2id:
            label2id = {str(k): int(v) for k, v in cfg.label2id.items()}
            id2label = {int(k): str(v) for k, v in cfg.id2label.items()}
            # Normalize keys that might be int-like strings already.
            return label2id, id2label
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not read labels from {base_model}: {exc}")
    return dict(DEFAULT_LABEL2ID), dict(DEFAULT_ID2LABEL)


def align_labels_with_tokenizer(examples, tokenizer, label2id, max_length: int):
    tokenized = tokenizer(
        examples["words"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    all_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                lab = labels[word_idx]
                label_ids.append(label2id.get(lab, label2id["0"]))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


def compute_metrics_builder(id2label: dict[int, str]):
    try:
        import evaluate

        seqeval = evaluate.load("seqeval")
    except Exception:  # noqa: BLE001
        seqeval = None

    def compute_metrics(p):
        predictions, labels = p
        preds = np.argmax(predictions, axis=2)
        true_predictions = []
        true_labels = []
        for pred_seq, lab_seq in zip(preds, labels):
            cur_p, cur_l = [], []
            for pred_i, lab_i in zip(pred_seq, lab_seq):
                if lab_i == -100:
                    continue
                cur_p.append(id2label[int(pred_i)])
                cur_l.append(id2label[int(lab_i)])
            true_predictions.append(cur_p)
            true_labels.append(cur_l)

        # Per-label accuracy (token-level, ignoring -100).
        flat_p = [x for seq in true_predictions for x in seq]
        flat_l = [x for seq in true_labels for x in seq]
        overall = float(np.mean([a == b for a, b in zip(flat_p, flat_l)])) if flat_l else 0.0

        per_label = {}
        for lab in PUNCT_LABELS:
            tp = sum(1 for a, b in zip(flat_p, flat_l) if a == lab and b == lab)
            fp = sum(1 for a, b in zip(flat_p, flat_l) if a == lab and b != lab)
            fn = sum(1 for a, b in zip(flat_p, flat_l) if a != lab and b == lab)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_label[f"f1_{lab}"] = f1

        # Macro-F1 over punctuation labels (exclude '0' which dominates).
        punct = [per_label[f"f1_{x}"] for x in PUNCT_LABELS if x != "0"]
        macro = float(np.mean(punct)) if punct else 0.0

        out = {"accuracy": overall, "f1": macro, **per_label}
        if seqeval is not None:
            # seqeval expects BIO-ish; our labels are already atomic tags.
            try:
                r = seqeval.compute(
                    predictions=[[f"B-{x}" if x != "0" else "O" for x in seq] for seq in true_predictions],
                    references=[[f"B-{x}" if x != "0" else "O" for x in seq] for seq in true_labels],
                )
                out["seqeval_f1"] = float(r["overall_f1"])
            except Exception:  # noqa: BLE001
                pass
        return out

    return compute_metrics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/train_kmr.yaml"))
    p.add_argument("--base-model", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-train-samples", type=int, default=None, help="Debug subset")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_model = args.base_model or cfg["base_model"]
    output_dir = Path(args.output_dir or cfg["output_dir"])
    set_seed(int(cfg.get("seed", 42)))

    label2id, id2label = build_label_maps(base_model)
    print("Labels:", label2id)

    train_rows = load_jsonl(Path(cfg["train_file"]))
    dev_rows = load_jsonl(Path(cfg["dev_file"]))
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]

    ds = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(dev_rows),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=False,
    )
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    max_length = int(cfg.get("max_length", 256))

    def _tok(batch):
        return align_labels_with_tokenizer(batch, tokenizer, label2id, max_length)

    tokenized = ds.map(_tok, batched=True, remove_columns=ds["train"].column_names)

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    ta_kwargs = dict(
        output_dir=str(output_dir),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 16)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 2)),
        num_train_epochs=float(cfg.get("num_train_epochs", 3)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.06)),
        fp16=bool(cfg.get("fp16", True)) and torch.cuda.is_available(),
        logging_steps=int(cfg.get("logging_steps", 50)),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 0)),
        report_to=[],
        save_total_limit=2,
        load_best_model_at_end=bool(cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=cfg.get("metric_for_best_model", "f1"),
        greater_is_better=bool(cfg.get("greater_is_better", True)),
        push_to_hub=bool(cfg.get("push_to_hub", False)),
        hub_model_id=cfg.get("hub_model_id"),
        hub_private_repo=bool(cfg.get("hub_private", True)),
    )
    # transformers 4.x: evaluation_strategy; 5.x: eval_strategy
    eval_strategy = cfg.get("eval_strategy", cfg.get("evaluation_strategy", "epoch"))
    save_strategy = cfg.get("save_strategy", "epoch")
    try:
        training_args = TrainingArguments(
            **ta_kwargs,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            eval_steps=cfg.get("eval_steps"),
            save_steps=cfg.get("save_steps"),
        )
    except TypeError:
        training_args = TrainingArguments(
            **ta_kwargs,
            evaluation_strategy=eval_strategy,
            save_strategy=save_strategy,
            eval_steps=cfg.get("eval_steps"),
            save_steps=cfg.get("save_steps"),
        )

    # transformers>=4.46 prefers processing_class; older versions want tokenizer=
    trainer_kw = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics_builder(id2label),
    )
    try:
        trainer = Trainer(**trainer_kw, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kw, tokenizer=tokenizer)

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval:", metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "label2id.json").write_text(json.dumps(label2id, indent=2), encoding="utf-8")
    print(f"Saved checkpoint → {output_dir}")

    if training_args.push_to_hub:
        trainer.push_to_hub(commit_message="Fine-tune FullStop for Kurmanji Latin (kmr)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
