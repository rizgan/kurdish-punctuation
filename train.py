#!/usr/bin/env python3
"""Train XLM-RoBERTa token classifier for Kurmanji punctuation."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset
from transformers import (
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_punctuation.constants import LABEL2ID, LABELS  # noqa: E402
from kurmanji_punctuation.dataset import label_distribution, load_processed_jsonl  # noqa: E402
from kurmanji_punctuation.label_alignment import align_labels_to_last_subtoken  # noqa: E402
from kurmanji_punctuation.model import bf16_supported, load_model, load_tokenizer  # noqa: E402
from kurmanji_punctuation.trainer import (  # noqa: E402
    WeightedPunctuationTrainer,
    build_compute_metrics,
    compute_class_weights,
    save_class_weights,
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


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
        label_ids = [LABEL2ID[x] for x in labs]
        all_labels.append(align_labels_to_last_subtoken(word_ids, label_ids))
    enc["labels"] = all_labels
    return enc


def make_training_args(cfg: dict, output_dir: Path) -> TrainingArguments:
    t = cfg["training"]
    use_bf16 = t.get("precision", "bf16") == "bf16" and bf16_supported()
    use_fp16 = (not use_bf16) and torch.cuda.is_available()

    common = dict(
        output_dir=str(output_dir),
        learning_rate=float(t["learning_rate"]),
        per_device_train_batch_size=int(t["train_batch_size"]),
        per_device_eval_batch_size=int(t["eval_batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        num_train_epochs=float(t["epochs"]),
        weight_decay=float(t["weight_decay"]),
        warmup_ratio=float(t["warmup_ratio"]),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        logging_steps=int(t["logging_steps"]),
        save_total_limit=int(t["save_total_limit"]),
        load_best_model_at_end=bool(t["load_best_model_at_end"]),
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=bool(t["greater_is_better"]),
        dataloader_num_workers=int(t.get("dataloader_num_workers", 0)),
        bf16=use_bf16,
        fp16=use_fp16 and not use_bf16,
        report_to=[],
        seed=int(cfg["project"]["seed"]),
    )
    eval_strategy = t.get("evaluation_strategy", t.get("eval_strategy", "epoch"))
    save_strategy = t.get("save_strategy", "epoch")
    try:
        return TrainingArguments(
            **common,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
        )
    except TypeError:
        return TrainingArguments(
            **common,
            evaluation_strategy=eval_strategy,
            save_strategy=save_strategy,
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--max-train-samples", type=int, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(cfg["project"]["seed"])
    set_all_seeds(seed)

    processed = Path(cfg.get("paths", {}).get("processed_dir", "data/processed"))
    output_dir = Path(cfg["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir = output_dir / "best"

    train_rows = load_processed_jsonl(processed / "train.jsonl")
    val_rows = load_processed_jsonl(processed / "validation.jsonl")
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]

    counts = label_distribution(train_rows)
    weights = None
    if cfg.get("loss", {}).get("use_class_weights", True):
        weights = compute_class_weights(
            counts,
            method=cfg["loss"].get("class_weight_method", "inverse_sqrt"),
            max_class_weight=float(cfg["loss"].get("max_class_weight", 8.0)),
        )
        save_class_weights(output_dir / "class_weights.json", weights, counts)

    tokenizer = load_tokenizer(cfg["model"]["name"])
    model = load_model(cfg["model"]["name"], num_labels=int(cfg["model"]["num_labels"]))
    if cfg["training"].get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    max_length = int(cfg["model"]["max_length"])
    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    def _tok(batch):
        return tokenize_batch(batch, tokenizer, max_length)

    train_tok = train_ds.map(_tok, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(_tok, batched=True, remove_columns=val_ds.column_names)

    training_args = make_training_args(cfg, output_dir)
    callbacks = []
    if cfg.get("early_stopping", {}).get("enabled", True):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(cfg["early_stopping"].get("patience", 2))
            )
        )

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer_kw = dict(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        compute_metrics=build_compute_metrics(),
        callbacks=callbacks,
        class_weights=weights,
    )
    try:
        trainer = WeightedPunctuationTrainer(**trainer_kw, processing_class=tokenizer)
    except TypeError:
        trainer = WeightedPunctuationTrainer(**trainer_kw, tokenizer=tokenizer)

    t0 = time.time()
    train_result = trainer.train()
    train_seconds = time.time() - t0
    metrics = trainer.evaluate()

    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    (best_dir / "labels.json").write_text(
        json.dumps({"labels": LABELS, "label2id": LABEL2ID}, indent=2), encoding="utf-8"
    )

    run_info = {
        "config": cfg,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": __import__("transformers").__version__,
        "n_train": len(train_rows),
        "n_validation": len(val_rows),
        "label_counts_train": counts,
        "class_weights": weights,
        "train_runtime_seconds": train_seconds,
        "train_loss": float(train_result.training_loss)
        if hasattr(train_result, "training_loss")
        else None,
        "eval_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics.items()},
        "best_metric": metrics.get(f"eval_{cfg['training']['metric_for_best_model']}", metrics.get(cfg["training"]["metric_for_best_model"])),
        "bf16": bool(training_args.bf16),
        "fp16": bool(training_args.fp16),
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    print(json.dumps(run_info["eval_metrics"], indent=2))
    print(f"Saved best model -> {best_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
