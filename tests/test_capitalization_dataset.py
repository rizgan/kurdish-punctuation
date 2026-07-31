"""Dataset labeling + case-only preservation."""

from kurmanji_capitalization.dataset import tokens_and_labels_from_article
from kurmanji_capitalization.casing import kurmanji_lower
from kurmanji_capitalization.sentence_rule import capitalize_sentence_starts
from kurmanji_capitalization.text_utils import validate_case_only_transformation


def test_sentence_starts_are_ignore():
    text = "Ez li Amedê dijîm. Navê min Azad e."
    tokens, labels, _ = tokens_and_labels_from_article(text)
    assert labels[0] == "IGNORE"  # Ez
    # Find Navê after period
    for tok, lab in zip(tokens, labels):
        if tok == "Navê":
            assert lab == "IGNORE"
            break


def test_proper_names_are_title():
    text = "Ez li Amedê dijîm. Navê min Azad e."
    tokens, labels, _ = tokens_and_labels_from_article(text)
    mapping = dict(zip(tokens, labels))
    assert mapping.get("amedê") == "TITLE"
    assert mapping.get("azad") == "TITLE"
    assert mapping.get("li") == "KEEP"


def test_punctuation_ignore():
    text = "Ez li Amedê dijîm."
    tokens, labels, _ = tokens_and_labels_from_article(text)
    for tok, lab in zip(tokens, labels):
        if tok in ".,?!":
            assert lab == "IGNORE"


def test_upper_acronym():
    text = "Ew ji UN tê."
    tokens, labels, _ = tokens_and_labels_from_article(text)
    mapping = {t: l for t, l in zip(tokens, labels)}
    assert mapping.get("un") == "UPPER" or mapping.get("UN") == "UPPER"


def test_case_only_preservation():
    inp = "Ez li amedê dijîm."
    out = "Ez li Amedê dijîm."
    assert validate_case_only_transformation(inp, out)
    assert not validate_case_only_transformation(inp, "Ez li Amedê dijim.")


def test_empty_and_number():
    assert tokens_and_labels_from_article("") is None
    triple = tokens_and_labels_from_article("2026.")
    assert triple is not None
