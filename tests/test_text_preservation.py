"""Text preservation: case, Kurmanji letters, URL, email, numbers."""

from kurmanji_punctuation.text_utils import (
    extract_words,
    join_words_with_punctuation,
    validate_text_preservation,
)


def test_case_unchanged():
    words = ["Ez", "ÇÛM", "Amedê"]
    out = join_words_with_punctuation(words, ["", ",", "."])
    assert extract_words(out) == words


def test_kurmanji_diacritics():
    words = ["çêşîû", "ÇÊŞÎÛ"]
    out = join_words_with_punctuation(words, ["", "."])
    assert "çêşîû" in out and "ÇÊŞÎÛ" in out
    assert validate_text_preservation("çêşîû ÇÊŞÎÛ", out)


def test_apostrophe_and_hyphen():
    words = ["navê", "welat-parêz"]
    out = join_words_with_punctuation(words, [",", "."])
    assert extract_words(out) == words


def test_url_email_numbers():
    words = ["binêre", "https://ku.wikipedia.org", "û", "name@example.com", "û", "42.5"]
    out = join_words_with_punctuation(words, ["", "", "", "", "", "."])
    assert extract_words(out) == words
    assert validate_text_preservation(" ".join(words), out)


def test_no_double_punctuation_in_join():
    out = join_words_with_punctuation(["a", "b"], [".", "."])
    assert ".." not in out.replace(". ", ".")
    # Explicit: each word gets at most one mark
    assert out == "a. b."


def test_single_token():
    out = join_words_with_punctuation(["Were"], ["!"])
    assert out == "Were!"
