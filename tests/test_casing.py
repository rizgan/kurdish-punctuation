"""Tests for Kurmanji casing transforms."""

from kurmanji_capitalization.casing import (
    apply_case_label,
    kurmanji_lower,
    kurmanji_title_token,
    kurmanji_upper,
)


def test_i_to_I_not_turkish():
    assert kurmanji_upper("i") == "I"
    assert "İ" not in kurmanji_upper("i")


def test_kurmanji_letters_upper():
    assert kurmanji_upper("î") == "Î"
    assert kurmanji_upper("ê") == "Ê"
    assert kurmanji_upper("ş") == "Ş"
    assert kurmanji_upper("ç") == "Ç"
    assert kurmanji_upper("û") == "Û"


def test_kurmanji_letters_lower():
    assert kurmanji_lower("Î") == "î"
    assert kurmanji_lower("Ê") == "ê"
    assert kurmanji_lower("Ş") == "ş"
    assert kurmanji_lower("Ç") == "ç"
    assert kurmanji_lower("Û") == "û"
    assert kurmanji_lower("I") == "i"
    assert kurmanji_lower("İ") == "i"


def test_title_amedê():
    assert kurmanji_title_token("amedê") == "Amedê"


def test_title_with_leading_quote():
    assert kurmanji_title_token("'amedê") == "'Amedê"


def test_title_leading_hyphen():
    assert kurmanji_title_token("-kurdistan") == "-Kurdistan"


def test_title_number_prefix():
    # First letter after digits is titled.
    assert kurmanji_title_token("2026an") == "2026An"


def test_apply_labels():
    assert apply_case_label("amedê", "TITLE") == "Amedê"
    assert apply_case_label("un", "UPPER") == "UN"
    assert apply_case_label("li", "KEEP") == "li"
