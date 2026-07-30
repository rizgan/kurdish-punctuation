"""Unicode / Kurmanji-safe text normalization (no case change, no transliteration)."""

from __future__ import annotations

import re
import unicodedata

from .constants import STRIP_PUNCT_CHARS, SUPPORTED_PUNCT

MULTISPACE_RE = re.compile(r"[ \t\f\v]+")
NEWLINE_RE = re.compile(r"\r\n|\r|\n")


def to_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    text = NEWLINE_RE.sub("\n", text)
    text = MULTISPACE_RE.sub(" ", text)
    # Collapse runs of blank lines to a single newline separator for article text.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_unsupported_punctuation(text: str, map_ellipsis_to_period: bool = True) -> str:
    """
    Remove punctuation that the model does not predict, without changing words.
    Ellipsis may become PERIOD for sentence splitting when configured.
    """
    if map_ellipsis_to_period:
        text = text.replace("...", ".")
        text = text.replace("…", ".")
    else:
        text = text.replace("...", " ")
        text = text.replace("…", " ")

    out: list[str] = []
    for ch in text:
        if ch in STRIP_PUNCT_CHARS:
            # Keep apostrophe/hyphen handling elsewhere; strip list may include quotes.
            out.append(" ")
            continue
        out.append(ch)
    return normalize_whitespace("".join(out))


def normalize_for_dataset(text: str, map_ellipsis_to_period: bool = True) -> str:
    """Full normalization pipeline for Wikipedia / corpus text before labeling."""
    text = to_nfc(text)
    text = strip_unsupported_punctuation(text, map_ellipsis_to_period=map_ellipsis_to_period)
    text = normalize_whitespace(text)
    return text


def normalize_for_inference(text: str) -> str:
    """Light normalize for inference input: NFC + whitespace only (keep existing ,.?!)."""
    text = to_nfc(text)
    text = normalize_whitespace(text)
    return text


def is_supported_punct(ch: str) -> bool:
    return ch in SUPPORTED_PUNCT
