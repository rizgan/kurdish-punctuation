"""Tokenization helpers and case-only preservation checks."""

from __future__ import annotations

import re

from .casing import kurmanji_lower, to_nfc
from .constants import SUPPORTED_PUNCT
from .normalization import normalize_whitespace

TOKEN_RE = re.compile(
    r"(?:"
    r"https?://[^\s]+|"
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}|"
    r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ſÇçÊêÎîŞşÛûİı][A-Za-zÀ-ÖØ-öø-ÿĀ-ſÇçÊêÎîŞşÛûİı0-9]*(?:['’\-][A-Za-zÀ-ÖØ-öø-ÿĀ-ſÇçÊêÎîŞşÛûİı0-9]+)*|"
    r"\d+(?:[.,]\d+)*|"
    r"[,.?!]|"
    r"[^\s]"
    r")",
    re.UNICODE,
)

URL_RE = re.compile(r"^https?://", re.IGNORECASE)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def tokenize_words_and_punct(text: str) -> list[str]:
    text = to_nfc(text)
    return [m.group(0) for m in TOKEN_RE.finditer(text)]


def is_url(token: str) -> bool:
    return bool(URL_RE.match(token))


def is_email(token: str) -> bool:
    return bool(EMAIL_RE.match(token))


def is_numeric_token(token: str) -> bool:
    return bool(token) and all(c.isdigit() or c in ".," for c in token) and any(c.isdigit() for c in token)


def validate_case_only_transformation(input_text: str, output_text: str) -> bool:
    """True iff after Kurmanji lowercasing both strings are identical."""
    left = kurmanji_lower(normalize_whitespace(to_nfc(input_text)))
    right = kurmanji_lower(normalize_whitespace(to_nfc(output_text)))
    return left == right


def assert_case_only(input_text: str, output_text: str) -> None:
    if not validate_case_only_transformation(input_text, output_text):
        raise RuntimeError(
            "Case-only preservation failed: output differs from input after kurmanji_lower()."
        )


def reconstruct_with_spacing(original: str, new_tokens: list[str]) -> str:
    """
    Replace token spellings in `original` while preserving whitespace and non-token chars.
    `new_tokens` must match TOKEN_RE finds in order.
    """
    original = to_nfc(original)
    parts: list[str] = []
    last = 0
    idx = 0
    for m in TOKEN_RE.finditer(original):
        parts.append(original[last : m.start()])
        if idx >= len(new_tokens):
            raise RuntimeError("Token count mismatch during reconstruction")
        parts.append(new_tokens[idx])
        idx += 1
        last = m.end()
    parts.append(original[last:])
    if idx != len(new_tokens):
        raise RuntimeError("Token count mismatch during reconstruction")
    return "".join(parts)
