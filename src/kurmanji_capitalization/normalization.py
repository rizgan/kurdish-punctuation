"""Unicode / Kurmanji-safe text normalization for capitalization corpus."""

from __future__ import annotations

import re

from .casing import to_nfc
from .constants import STRIP_PUNCT_CHARS, SUPPORTED_PUNCT

MULTISPACE_RE = re.compile(r"[ \t\f\v]+")
NEWLINE_RE = re.compile(r"\r\n|\r|\n")


def normalize_whitespace(text: str) -> str:
    text = NEWLINE_RE.sub("\n", text)
    text = MULTISPACE_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_unsupported_punctuation(text: str, map_ellipsis_to_period: bool = True) -> str:
    if map_ellipsis_to_period:
        text = text.replace("...", ".")
        text = text.replace("…", ".")
    else:
        text = text.replace("...", " ")
        text = text.replace("…", " ")

    out: list[str] = []
    for ch in text:
        if ch in STRIP_PUNCT_CHARS:
            out.append(" ")
            continue
        out.append(ch)
    return normalize_whitespace("".join(out))


def normalize_for_dataset(text: str, map_ellipsis_to_period: bool = True) -> str:
    text = to_nfc(text)
    text = strip_unsupported_punctuation(text, map_ellipsis_to_period=map_ellipsis_to_period)
    text = normalize_whitespace(text)
    return text


def normalize_for_inference(text: str) -> str:
    text = to_nfc(text)
    text = normalize_whitespace(text)
    return text


def is_supported_punct(ch: str) -> bool:
    return ch in SUPPORTED_PUNCT
