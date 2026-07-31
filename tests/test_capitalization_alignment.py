"""Capitalization labels go on the *first* subtoken (unlike punctuation)."""

from kurmanji_punctuation.label_alignment import align_labels_to_last_subtoken
from kurmanji_capitalization.label_alignment import align_labels_to_first_subtoken


def test_first_subtoken_gets_label():
    word_ids = [None, 0, 0, 1, None]
    labels = [1, 2]  # TITLE, UPPER
    aligned = align_labels_to_first_subtoken(word_ids, labels)
    assert aligned == [-100, 1, -100, 2, -100]


def test_remaining_subtokens_ignored():
    word_ids = [0, 0, 0]
    labels = [2]
    aligned = align_labels_to_first_subtoken(word_ids, labels)
    assert aligned == [2, -100, -100]


def test_contrast_with_punctuation_last_subtoken():
    word_ids = [None, 0, 0, 1, None]
    labels = [1, 2]
    first = align_labels_to_first_subtoken(word_ids, labels)
    last = align_labels_to_last_subtoken(word_ids, labels)
    assert first == [-100, 1, -100, 2, -100]
    assert last == [-100, -100, 1, 2, -100]
    assert first != last
