"""Tokenization helpers and text-preservation checks."""

from __future__ import annotations

import re

from .constants import SUPPORTED_PUNCT
from .normalization import normalize_whitespace, to_nfc

# Words keep internal apostrophes and hyphens; digits may attach to letters (word42).
TOKEN_RE = re.compile(
    r"(?:"
    r"https?://[^\s]+|"
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}|"
    r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ſÇçÊêÎîŞşÛû][A-Za-zÀ-ÖØ-öø-ÿĀ-ſÇçÊêÎîŞşÛû0-9]*(?:['’\-][A-Za-zÀ-ÖØ-öø-ÿĀ-ſÇçÊêÎîŞşÛû0-9]+)*|"
    r"\d+(?:[.,]\d+)*|"
    r"[,.?!]|"
    r"[^\s]"
    r")",
    re.UNICODE,
)


def tokenize_words_and_punct(text: str) -> list[str]:
    text = to_nfc(text)
    return [m.group(0) for m in TOKEN_RE.finditer(text)]


def extract_words(text: str) -> list[str]:
    """Return non-punctuation tokens in order (preserves case / Kurmanji letters)."""
    toks = tokenize_words_and_punct(text)
    return [t for t in toks if t not in SUPPORTED_PUNCT]


def join_words_with_punctuation(tokens: list[str], punct_after: list[str]) -> str:
    """
    tokens: words only
    punct_after: '' or one of , . ? ! for each word
    """
    if len(tokens) != len(punct_after):
        raise ValueError("tokens and punct_after length mismatch")
    parts: list[str] = []
    for i, (word, punct) in enumerate(zip(tokens, punct_after)):
        piece = word + (punct if punct in SUPPORTED_PUNCT else "")
        parts.append(piece)
        if i < len(tokens) - 1:
            parts.append(" ")
    return "".join(parts)


def strip_supported_punctuation(text: str) -> str:
    chars = []
    for ch in text:
        if ch in SUPPORTED_PUNCT:
            continue
        chars.append(ch)
    return normalize_whitespace("".join(chars))


def validate_text_preservation(input_text: str, output_text: str) -> bool:
    """
    True iff output equals input after removing only inserted , . ? ! and normalizing spaces.
    """
    inp = normalize_whitespace(to_nfc(input_text))
    out = normalize_whitespace(to_nfc(output_text))
    # Compare word/char sequence with supported punct stripped from both sides.
    # Input may already contain supported punct — strip from both for word identity.
    left = strip_supported_punctuation(inp)
    right = strip_supported_punctuation(out)
    return left == right


def words_equal(a: list[str], b: list[str]) -> bool:
    return a == b
