"""Tests for last-subtoken label alignment."""

from kurmanji_punctuation.label_alignment import align_labels_to_last_subtoken


def test_single_subtoken_word():
    # [CLS]=None, word0, [SEP]=None
    word_ids = [None, 0, None]
    labels = [3]
    aligned = align_labels_to_last_subtoken(word_ids, labels)
    assert aligned == [-100, 3, -100]


def test_multi_subtoken_word():
    # word 0 -> two pieces; label on last
    word_ids = [None, 0, 0, 1, None]
    labels = [1, 2]  # COMMA, PERIOD
    aligned = align_labels_to_last_subtoken(word_ids, labels)
    assert aligned == [-100, -100, 1, 2, -100]


def test_special_tokens_are_ignored():
    word_ids = [None, None]
    aligned = align_labels_to_last_subtoken(word_ids, [])
    assert aligned == [-100, -100]


def test_padding_none():
    word_ids = [None, 0, None, None]
    labels = [4]
    aligned = align_labels_to_last_subtoken(word_ids, labels)
    assert aligned == [-100, 4, -100, -100]


def test_repeated_word_ids():
    word_ids = [0, 0, 0]
    labels = [2]
    aligned = align_labels_to_last_subtoken(word_ids, labels)
    assert aligned == [-100, -100, 2]


def test_last_word_in_sequence():
    word_ids = [0, 1, 1]
    labels = [0, 3]
    aligned = align_labels_to_last_subtoken(word_ids, labels)
    assert aligned == [0, -100, 3]
