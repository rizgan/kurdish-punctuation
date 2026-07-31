"""Capitalization label scheme (v1: KEEP / TITLE / UPPER)."""

from __future__ import annotations

LABELS = [
    "KEEP",
    "TITLE",
    "UPPER",
]

# Stored in JSONL; mapped to -100 before training.
IGNORE_LABEL = "IGNORE"

TRAINABLE_LABELS = LABELS  # KEEP included in CE; macro F1 excludes it

MACRO_F1_LABELS = ["TITLE", "UPPER"]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

SUPPORTED_PUNCT = frozenset(",.?!")
SENTENCE_END_PUNCT = frozenset(".?!")

# Removed during corpus prep (not predicted).
STRIP_PUNCT_CHARS = frozenset(":;—–…\"'«»“”()[]{}<>/")
