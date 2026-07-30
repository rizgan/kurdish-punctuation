"""Inference helpers compatible with deepmultilingualpunctuation.PunctuationModel."""

from __future__ import annotations

import re
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


class KurdishPunctuationModel:
    """
    FullStop-compatible punctuation restoration.

    A fine-tuned checkpoint can also be loaded with:
      from deepmultilingualpunctuation import PunctuationModel
      PunctuationModel(model="checkpoints/kmr-fullstop")
    """

    def __init__(self, model: str = "checkpoints/kmr-fullstop", device: int | str | None = None) -> None:
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.model_name = model
        self._pipe = pipeline(
            "token-classification",
            model=model,
            aggregation_strategy="none",
            device=device,
        )

    def preprocess(self, text: str) -> str:
        # Remove markers except decimal points inside numbers (FullStop-style).
        text = re.sub(r"(?<!\d)[.,;:!?\-—…](?!\d)", " ", text)
        text = text.replace("؟", " ").replace("،", " ")
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    def predict(self, text: str) -> list[list[Any]]:
        clean = self.preprocess(text)
        if not clean:
            return []

        words = clean.split()
        # Char offsets of each whitespace-separated word in `clean`.
        spans: list[tuple[int, int]] = []
        pos = 0
        for w in words:
            start = clean.find(w, pos)
            end = start + len(w)
            spans.append((start, end))
            pos = end

        outs = self._pipe(clean)
        labeled: list[list[Any]] = []
        for (start, end), w in zip(spans, words):
            best_label = "0"
            best_score = 0.0
            zero_score = 0.0
            for tok in outs:
                ts, te = int(tok["start"]), int(tok["end"])
                if te <= start or ts >= end:
                    continue
                raw = tok.get("entity_group") or tok.get("entity") or "0"
                label = str(raw).replace("B-", "").replace("I-", "").replace("LABEL_", "")
                score = float(tok.get("score", 0.0))
                if label == "0":
                    zero_score = max(zero_score, score)
                elif score >= best_score:
                    best_label = label
                    best_score = score
            labeled.append([w, best_label, best_score if best_label != "0" else zero_score])
        return labeled

    def restore_punctuation(self, text: str) -> str:
        labeled = self.predict(text)
        if not labeled:
            return text
        parts: list[str] = []
        capitalize_next = True
        for word, label, _score in labeled:
            w = word
            if capitalize_next and w:
                w = w[0].upper() + w[1:]
                capitalize_next = False
            if label in {".", "?"}:
                parts.append(w + label)
                capitalize_next = True
            elif label in {",", "-", ":"}:
                parts.append(w + label)
            else:
                parts.append(w)
        out = " ".join(parts)
        out = re.sub(r"\s+([.,?:;\-])", r"\1", out)
        return out.strip()


def load_model_and_tokenizer(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    return model, tokenizer
