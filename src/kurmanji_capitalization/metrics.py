"""Metrics for capitalization (macro F1 over TITLE+UPPER only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import ID2LABEL, LABELS, MACRO_F1_LABELS


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def compute_capitalization_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, Any]:
    assert len(y_true) == len(y_pred)
    per_label: dict[str, dict[str, float]] = {}
    for lab in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec, rec, f1 = _prf(tp, fp, fn)
        support = sum(1 for t in y_true if t == lab)
        per_label[lab] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": float(support),
        }

    macro = float(np.mean([per_label[l]["f1"] for l in MACRO_F1_LABELS])) if MACRO_F1_LABELS else 0.0

    # Micro over TITLE+UPPER only
    focus = set(MACRO_F1_LABELS)
    tp_m = sum(1 for t, p in zip(y_true, y_pred) if t == p and t in focus)
    fp_m = sum(1 for t, p in zip(y_true, y_pred) if p in focus and t != p)
    fn_m = sum(1 for t, p in zip(y_true, y_pred) if t in focus and t != p)
    micro_p, micro_r, micro_f1 = _prf(tp_m, fp_m, fn_m)

    weighted_num = 0.0
    weighted_den = 0.0
    for lab in LABELS:
        f1 = per_label[lab]["f1"]
        support = per_label[lab]["support"]
        weighted_num += f1 * support
        weighted_den += support
    weighted_f1 = weighted_num / weighted_den if weighted_den else 0.0
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(1, len(y_true))

    # Proper-name = TITLE; acronym = UPPER
    proper_name_f1 = per_label["TITLE"]["f1"]
    acronym_f1 = per_label["UPPER"]["f1"]

    index = {lab: i for i, lab in enumerate(LABELS)}
    mat = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            mat[index[t], index[p]] += 1

    return {
        "capitalization_macro_f1": macro,
        "capitalization_micro_f1": micro_f1,
        "capitalization_micro_precision": micro_p,
        "capitalization_micro_recall": micro_r,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "proper_name_f1": proper_name_f1,
        "proper_name_precision": per_label["TITLE"]["precision"],
        "proper_name_recall": per_label["TITLE"]["recall"],
        "acronym_f1": acronym_f1,
        "per_label": per_label,
        "confusion_matrix": mat.tolist(),
        "labels": list(LABELS),
    }


def flatten_predictions(logits, labels) -> tuple[list[str], list[str]]:
    """Drop -100 positions; map ids to label strings."""
    if hasattr(logits, "numpy"):
        logits = logits.numpy()
    if hasattr(labels, "numpy"):
        labels = labels.numpy()
    preds = np.argmax(logits, axis=-1)
    y_true: list[str] = []
    y_pred: list[str] = []
    for pred_row, lab_row in zip(preds, labels):
        for p, t in zip(pred_row, lab_row):
            if int(t) == -100:
                continue
            y_true.append(ID2LABEL[int(t)])
            y_pred.append(ID2LABEL[int(p)])
    return y_true, y_pred
