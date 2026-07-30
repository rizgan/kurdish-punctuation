"""Punctuation label scheme for Kurmanji restoration."""

from __future__ import annotations

LABELS = [
    "O",
    "COMMA",
    "PERIOD",
    "QUESTION",
    "EXCLAMATION",
]

PUNCTUATION_TO_LABEL = {
    ",": "COMMA",
    ".": "PERIOD",
    "?": "QUESTION",
    "!": "EXCLAMATION",
}

LABEL_TO_PUNCTUATION = {
    "O": "",
    "COMMA": ",",
    "PERIOD": ".",
    "QUESTION": "?",
    "EXCLAMATION": "!",
}

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

SUPPORTED_PUNCT = frozenset(PUNCTUATION_TO_LABEL.keys())
SENTENCE_END_PUNCT = frozenset(".?!")
SENTENCE_BOUNDARY_LABELS = frozenset({"PERIOD", "QUESTION", "EXCLAMATION"})

# Removed during corpus prep (not predicted in v1).
STRIP_PUNCT_CHARS = frozenset(":;—–…\"'«»“”()[]{}<>/")
