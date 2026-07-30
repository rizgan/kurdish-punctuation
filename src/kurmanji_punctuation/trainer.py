"""Custom Trainer with inverse-sqrt class weights."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import Trainer

from .constants import ID2LABEL, LABEL2ID, LABELS


def compute_class_weights(
    label_counts: dict[str, int],
    *,
    method: str = "inverse_sqrt",
    max_class_weight: float = 8.0,
) -> list[float]:
    """
    weight[c] = 1 / sqrt(freq[c]), then normalize so weight[O] == 1,
    then clip to max_class_weight.
    """
    if method != "inverse_sqrt":
        raise ValueError(f"Unsupported class_weight_method: {method}")

    freqs = []
    for lab in LABELS:
        freqs.append(max(1, int(label_counts.get(lab, 0))))

    raw = [1.0 / math.sqrt(f) for f in freqs]
    o_idx = LABEL2ID["O"]
    scale = raw[o_idx]
    weights = [w / scale for w in raw]
    weights = [min(max_class_weight, w) for w in weights]
    return weights


class WeightedPunctuationTrainer(Trainer):
    def __init__(self, *args, class_weights: list[float] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits  # [B, T, C]
        if labels is None:
            loss = outputs.loss
        else:
            if self._class_weights is not None:
                weight = torch.tensor(
                    self._class_weights,
                    device=logits.device,
                    dtype=logits.dtype,
                )
                loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def save_class_weights(path: Path, weights: list[float], counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": LABELS,
        "weights": {lab: weights[i] for i, lab in enumerate(LABELS)},
        "weights_list": weights,
        "label_counts": counts,
        "method": "inverse_sqrt_normalized_O_equals_1",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_compute_metrics():
    from .metrics import compute_punctuation_metrics, flatten_predictions

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        y_true, y_pred = flatten_predictions(logits, labels)
        m = compute_punctuation_metrics(y_true, y_pred)
        out = {
            "punctuation_macro_f1": m["punctuation_macro_f1"],
            "punctuation_micro_f1": m["punctuation_micro_f1"],
            "sentence_boundary_f1": m["sentence_boundary_f1"],
            "weighted_f1": m["weighted_f1"],
            "accuracy": m["accuracy"],
        }
        for lab, d in m["per_label"].items():
            out[f"f1_{lab}"] = d["f1"]
            out[f"precision_{lab}"] = d["precision"]
            out[f"recall_{lab}"] = d["recall"]
        return out

    return compute_metrics
