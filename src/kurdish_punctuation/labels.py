"""Label scheme aligned with oliverguhr FullStop models."""

from __future__ import annotations

# Punctuation restored by FullStop (word-level: label = mark after the word).
PUNCT_LABELS = ("0", ".", ",", "?", "-", ":")

# Characters treated as punctuation targets when building training examples.
# "!" and ";" are mapped to closest FullStop labels for sparse wiki text.
PUNCT_CHARS = frozenset(".?,:;-!")

PUNCT_NORMALIZE = {
    "!": ".",
    ";": ",",
    "…": ".",
    "؟": "?",  # Arabic question mark — should be rare after Latin filter
    "،": ",",  # Arabic comma
}

DEFAULT_LABEL2ID = {label: i for i, label in enumerate(PUNCT_LABELS)}
DEFAULT_ID2LABEL = {i: label for label, i in DEFAULT_LABEL2ID.items()}


def normalize_punct_char(ch: str) -> str | None:
    """Map a punctuation character to a FullStop label, or None if ignored."""
    if ch in PUNCT_NORMALIZE:
        ch = PUNCT_NORMALIZE[ch]
    if ch in PUNCT_LABELS and ch != "0":
        return ch
    return None
