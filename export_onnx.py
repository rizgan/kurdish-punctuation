#!/usr/bin/env python3
"""Optional ONNX export for the trained punctuation model."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_punctuation.constants import LABEL2ID, LABELS  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="PyTorch checkpoint dir")
    p.add_argument("--out-dir", type=Path, default=Path("onnx"))
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.eval()

    # Dummy batch for export tracing
    enc = tokenizer(
        ["ez", "îro", "çûm"],
        is_split_into_words=True,
        return_tensors="pt",
        padding="max_length",
        max_length=32,
        truncation=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    onnx_path = args.out_dir / "model.onnx"

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapped = Wrapper(model)
    torch.onnx.export(
        wrapped,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=args.opset,
    )

    tokenizer.save_pretrained(args.out_dir)
    # Copy HF config if present
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"):
        src = args.model / name
        if src.exists():
            shutil.copy2(src, args.out_dir / name)
    (args.out_dir / "labels.json").write_text(
        json.dumps({"labels": LABELS, "label2id": LABEL2ID}, indent=2), encoding="utf-8"
    )
    print(f"Exported ONNX -> {onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
