"""Tests for deterministic sentence-start capitalization."""

from kurmanji_capitalization.casing import kurmanji_lower
from kurmanji_capitalization.sentence_rule import capitalize_sentence_starts


def test_start_of_text():
    assert capitalize_sentence_starts("ez li vir im.") == "Ez li vir im."


def test_after_period():
    assert capitalize_sentence_starts("ez li vir im. ew li wir e.") == "Ez li vir im. Ew li wir e."


def test_after_question():
    assert capitalize_sentence_starts("tu kî yî? ez azad im.") == "Tu kî yî? Ez azad im."


def test_after_exclamation():
    assert capitalize_sentence_starts("were! ez hatim.") == "Were! Ez hatim."


def test_quote_after_period():
    out = capitalize_sentence_starts('got. "ez tême."')
    assert '"Ez' in out


def test_pipeline_lower_then_rule():
    text = "ez li amedê dijîm. navê min azad e."
    out = capitalize_sentence_starts(kurmanji_lower(text))
    assert out.startswith("Ez ")
    assert ". Navê " in out or ".Navê " in out
