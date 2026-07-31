"""Kurmanji-safe case transforms (never use Turkish i→İ)."""

from __future__ import annotations

import re
import unicodedata

# Explicit maps — do not rely on locale / str.upper for Kurmanji letters.
_LOWER_TO_UPPER = {
    "a": "A",
    "b": "B",
    "c": "C",
    "ç": "Ç",
    "d": "D",
    "e": "E",
    "ê": "Ê",
    "f": "F",
    "g": "G",
    "h": "H",
    "i": "I",
    "î": "Î",
    "j": "J",
    "k": "K",
    "l": "L",
    "m": "M",
    "n": "N",
    "o": "O",
    "p": "P",
    "q": "Q",
    "r": "R",
    "s": "S",
    "ş": "Ş",
    "t": "T",
    "u": "U",
    "û": "Û",
    "v": "V",
    "w": "W",
    "x": "X",
    "y": "Y",
    "z": "Z",
    # Latin extras that may appear in loanwords / wiki
    "ä": "Ä",
    "ö": "Ö",
    "ü": "Ü",
    "á": "Á",
    "à": "À",
    "é": "É",
    "è": "È",
    "ó": "Ó",
    "ò": "Ò",
}

# Also map Turkish İ → i when lowering (wiki sometimes has Turkish casing).
_UPPER_TO_LOWER = {u: lo for lo, u in _LOWER_TO_UPPER.items()}
_UPPER_TO_LOWER["İ"] = "i"
_UPPER_TO_LOWER["I"] = "i"

_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def to_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def is_letter(ch: str) -> bool:
    return bool(ch) and unicodedata.category(ch).startswith("L")


def kurmanji_lower_char(ch: str) -> str:
    ch = to_nfc(ch)
    if ch in _UPPER_TO_LOWER:
        return _UPPER_TO_LOWER[ch]
    if len(ch) == 1 and ch.isupper():
        # Fallback for uncommon Latin letters — still avoid Turkish dotted I.
        if ch == "İ":
            return "i"
        return ch.lower()
    return ch


def kurmanji_upper_char(ch: str) -> str:
    ch = to_nfc(ch)
    if ch in _LOWER_TO_UPPER:
        return _LOWER_TO_UPPER[ch]
    if len(ch) == 1 and ch.islower():
        if ch == "i":
            return "I"
        return ch.upper()
    return ch


def kurmanji_lower(text: str) -> str:
    return "".join(kurmanji_lower_char(c) for c in to_nfc(text))


def kurmanji_upper(text: str) -> str:
    return "".join(kurmanji_upper_char(c) for c in to_nfc(text))


def first_letter_index(token: str) -> int | None:
    for i, ch in enumerate(token):
        if is_letter(ch):
            return i
    return None


def kurmanji_title_token(token: str) -> str:
    """Uppercase first letter only; leave remaining letters lowercase."""
    token = to_nfc(token)
    idx = first_letter_index(token)
    if idx is None:
        return token
    chars = list(token)
    chars[idx] = kurmanji_upper_char(chars[idx])
    for j in range(idx + 1, len(chars)):
        if is_letter(chars[j]):
            chars[j] = kurmanji_lower_char(chars[j])
    return "".join(chars)


def letter_chars(token: str) -> list[str]:
    return [c for c in to_nfc(token) if is_letter(c)]


def is_all_upper_letters(token: str) -> bool:
    letters = letter_chars(token)
    return len(letters) >= 2 and all(kurmanji_upper_char(kurmanji_lower_char(c)) == c for c in letters)


def is_title_case_letters(token: str) -> bool:
    """First letter upper, remaining letters lower (non-letters ignored)."""
    letters = letter_chars(token)
    if not letters:
        return False
    first_ok = kurmanji_upper_char(kurmanji_lower_char(letters[0])) == letters[0]
    rest_ok = all(kurmanji_lower_char(c) == c for c in letters[1:])
    return first_ok and rest_ok


def is_all_lower_letters(token: str) -> bool:
    letters = letter_chars(token)
    return bool(letters) and all(kurmanji_lower_char(c) == c for c in letters)


def apply_case_label(token: str, label: str) -> str:
    if label == "KEEP":
        return token
    if label == "TITLE":
        return kurmanji_title_token(token)
    if label == "UPPER":
        return kurmanji_upper(token)
    raise ValueError(f"Unknown case label: {label}")
