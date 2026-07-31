"""Deterministic sentence-start capitalization (not learned by XLM-R)."""

from __future__ import annotations

from dataclasses import dataclass

from .casing import first_letter_index, kurmanji_title_token, to_nfc
from .constants import SENTENCE_END_PUNCT, SUPPORTED_PUNCT
from .text_utils import reconstruct_with_spacing, tokenize_words_and_punct


@dataclass(frozen=True)
class SentenceRuleConfig:
    capitalize_first_word: bool = True
    capitalize_after_period: bool = True
    capitalize_after_question: bool = True
    capitalize_after_exclamation: bool = True
    skip_leading_quotes: bool = True
    capitalize_word_after_leading_number: bool = False


SKIP_TOKENS = SUPPORTED_PUNCT | frozenset("\"'«»“”‘’()[]{}<>")


def _should_trigger(ch: str, cfg: SentenceRuleConfig) -> bool:
    if ch == "." and cfg.capitalize_after_period:
        return True
    if ch == "?" and cfg.capitalize_after_question:
        return True
    if ch == "!" and cfg.capitalize_after_exclamation:
        return True
    return False


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


def capitalize_sentence_starts(
    text: str,
    cfg: SentenceRuleConfig | None = None,
) -> str:
    """
    Title-case the first letter-word of the text and after standalone . ? ! tokens.

    Token-based (not character-based) so periods inside URLs, emails, and decimals
    do not trigger a new sentence.
    """
    cfg = cfg or SentenceRuleConfig()
    text = to_nfc(text)
    if not text:
        return text

    tokens = tokenize_words_and_punct(text)
    if not tokens:
        return text

    starts = sentence_start_word_indices(tokens, cfg)
    new_tokens = [
        kurmanji_title_token(tok) if i in starts else tok for i, tok in enumerate(tokens)
    ]
    return reconstruct_with_spacing(text, new_tokens)
