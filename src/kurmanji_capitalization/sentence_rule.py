"""Deterministic sentence-start capitalization (not learned by XLM-R)."""

from __future__ import annotations

from dataclasses import dataclass

from .casing import first_letter_index, is_letter, kurmanji_title_token, to_nfc
from .constants import SENTENCE_END_PUNCT, SUPPORTED_PUNCT


@dataclass(frozen=True)
class SentenceRuleConfig:
    capitalize_first_word: bool = True
    capitalize_after_period: bool = True
    capitalize_after_question: bool = True
    capitalize_after_exclamation: bool = True
    skip_leading_quotes: bool = True
    capitalize_word_after_leading_number: bool = False


LEADING_SKIP = frozenset("\"'«»“”‘’()[]{}<> ")
SKIP_TOKENS = SUPPORTED_PUNCT | frozenset("\"'«»“”‘’()[]{}<>")


def _should_trigger(ch: str, cfg: SentenceRuleConfig) -> bool:
    if ch == "." and cfg.capitalize_after_period:
        return True
    if ch == "?" and cfg.capitalize_after_question:
        return True
    if ch == "!" and cfg.capitalize_after_exclamation:
        return True
    return False


def capitalize_sentence_starts(
    text: str,
    cfg: SentenceRuleConfig | None = None,
) -> str:
    """
    Title-case the first letter-word of the text and after . ? ! .
    Does not alter other letters in the token; skips quotes/brackets between
    terminator and the next word.
    """
    cfg = cfg or SentenceRuleConfig()
    text = to_nfc(text)
    if not text:
        return text

    chars = list(text)
    n = len(chars)
    capitalize_next = bool(cfg.capitalize_first_word)
    i = 0
    while i < n:
        ch = chars[i]
        if capitalize_next:
            if cfg.skip_leading_quotes and ch in LEADING_SKIP:
                i += 1
                continue
            if ch.isdigit():
                while i < n and (chars[i].isdigit() or chars[i] in ".,"):
                    i += 1
                if not cfg.capitalize_word_after_leading_number:
                    capitalize_next = False
                continue
            if is_letter(ch):
                j = i
                while j < n and (
                    is_letter(chars[j])
                    or chars[j].isdigit()
                    or chars[j] in "'’-"
                ):
                    j += 1
                token = "".join(chars[i:j])
                titled = kurmanji_title_token(token)
                chars[i:j] = list(titled)
                capitalize_next = False
                i = j
                continue
            i += 1
            continue

        if ch in SENTENCE_END_PUNCT and _should_trigger(ch, cfg):
            capitalize_next = True
        i += 1

    return "".join(chars)


def sentence_start_word_indices(
    tokens: list[str],
    cfg: SentenceRuleConfig | None = None,
) -> set[int]:
    """
    Indices into `tokens` (words + punct) that are sentence-start letter-words
    handled by the deterministic rule → IGNORE for the model.
    """
    cfg = cfg or SentenceRuleConfig()
    starts: set[int] = set()
    capitalize_next = bool(cfg.capitalize_first_word)
    for i, tok in enumerate(tokens):
        if tok in SENTENCE_END_PUNCT and _should_trigger(tok, cfg):
            capitalize_next = True
            continue
        if tok in SKIP_TOKENS:
            continue
        if first_letter_index(tok) is None:
            if any(c.isdigit() for c in tok) and capitalize_next:
                if not cfg.capitalize_word_after_leading_number:
                    capitalize_next = False
            continue
        if capitalize_next:
            starts.add(i)
            capitalize_next = False
    return starts
