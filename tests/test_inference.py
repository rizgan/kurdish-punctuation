"""Inference tests that do not require a trained checkpoint (logic only)."""

from kurmanji_punctuation.inference import PunctuationRestorer
from kurmanji_punctuation.text_utils import (
    extract_words,
    join_words_with_punctuation,
    validate_text_preservation,
)


def test_empty_text_join():
    assert join_words_with_punctuation([], []) == ""


def test_spacing_rules():
    out = join_words_with_punctuation(["ez", "hatim"], ["", "."])
    assert out == "ez hatim."
    out2 = join_words_with_punctuation(["a", "b", "c"], [",", "", "?"])
    assert out2 == "a, b c?"


def test_no_space_before_punct():
    out = join_words_with_punctuation(["belê"], [","])
    assert " ," not in out
    assert out.endswith(",")


def test_validate_preserves_words():
    inp = "ez îro çûm bajarê"
    out = "ez îro çûm bajarê."
    assert validate_text_preservation(inp, out)


def test_long_word_list_no_loss():
    words = [f"word{i}" for i in range(10_001)]
    punct = [""] * len(words)
    punct[-1] = "."
    text = join_words_with_punctuation(words, punct)
    recovered = extract_words(text)
    assert recovered == words
    assert len(recovered) == 10_001
