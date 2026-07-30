"""Evaluation metrics focused on punctuation classes (exclude O from macro)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .constants import ID2LABEL, LABELS, SENTENCE_BOUNDARY_LABELS


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def compute_punctuation_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, Any]:
    assert len(y_true) == len(y_pred)
    punct_labels = [lab for lab in LABELS if lab != "O"]
    per_label: dict[str, dict[str, float]] = {}
    for lab in punct_labels:
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

    macro = float(np.mean([per_label[l]["f1"] for l in punct_labels])) if punct_labels else 0.0

    # Micro over punctuation only (treat non-punct as outside).
    tp_m = sum(1 for t, p in zip(y_true, y_pred) if t == p and t != "O")
    fp_m = sum(1 for t, p in zip(y_true, y_pred) if p != "O" and t != p)
    fn_m = sum(1 for t, p in zip(y_true, y_pred) if t != "O" and t != p)
    micro_p, micro_r, micro_f1 = _prf(tp_m, fp_m, fn_m)

    # Weighted F1 over all labels including O.
    weighted_num = 0.0
    weighted_den = 0.0
    for lab in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        _p, _r, f1 = _prf(tp, fp, fn)
        support = sum(1 for t in y_true if t == lab)
        weighted_num += f1 * support
        weighted_den += support
    weighted_f1 = weighted_num / weighted_den if weighted_den else 0.0

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(1, len(y_true))

    # Sentence boundary: merge PERIOD/QUESTION/EXCLAMATION.
    def to_boundary(lab: str) -> str:
        return "BOUNDARY" if lab in SENTENCE_BOUNDARY_LABELS else "OTHER"

    bt = [to_boundary(x) for x in y_true]
    bp = [to_boundary(x) for x in y_pred]
    tp_b = sum(1 for t, p in zip(bt, bp) if t == "BOUNDARY" and p == "BOUNDARY")
    fp_b = sum(1 for t, p in zip(bt, bp) if t != "BOUNDARY" and p == "BOUNDARY")
    fn_b = sum(1 for t, p in zip(bt, bp) if t == "BOUNDARY" and p != "BOUNDARY")
    _bp, _br, boundary_f1 = _prf(tp_b, fp_b, fn_b)

    # Confusion matrix (rows=true, cols=pred)
    index = {lab: i for i, lab in enumerate(LABELS)}
    mat = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            mat[index[t], index[p]] += 1

    return {
        "per_label": per_label,
        "punctuation_macro_f1": macro,
        "punctuation_micro_f1": micro_f1,
        "weighted_f1": float(weighted_f1),
        "accuracy": float(accuracy),
        "sentence_boundary_f1": float(boundary_f1),
        "confusion_matrix": mat.tolist(),
        "labels": list(LABELS),
        "support": dict(Counter(y_true)),
    }


def flatten_predictions(
    predictions: np.ndarray,
    label_ids: np.ndarray,
) -> tuple[list[str], list[str]]:
    """Argmax predictions vs labels, skipping -100."""
    preds = np.argmax(predictions, axis=-1)
    y_true: list[str] = []
    y_pred: list[str] = []
    for pred_row, lab_row in zip(preds, label_ids):
        for p, l in zip(pred_row, lab_row):
            if int(l) == -100:
                continue
            y_true.append(ID2LABEL[int(l)])
            y_pred.append(ID2LABEL[int(p)])
    return y_true, y_pred


def format_classification_report(metrics: dict[str, Any]) -> str:
    lines = [
        f"punctuation_macro_f1: {metrics['punctuation_macro_f1']:.4f}",
        f"punctuation_micro_f1: {metrics['punctuation_micro_f1']:.4f}",
        f"sentence_boundary_f1: {metrics['sentence_boundary_f1']:.4f}",
        f"weighted_f1: {metrics['weighted_f1']:.4f}",
        f"accuracy: {metrics['accuracy']:.4f}",
        "",
        f"{'label':<14} {'P':>8} {'R':>8} {'F1':>8} {'support':>10}",
    ]
    for lab, d in metrics["per_label"].items():
        lines.append(
            f"{lab:<14} {d['precision']:8.4f} {d['recall']:8.4f} {d['f1']:8.4f} {int(d['support']):10d}"
        )
    return "\n".join(lines) + "\n"
