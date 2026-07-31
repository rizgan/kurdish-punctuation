"""Tests for dataset labeling, article splits, and word windows."""

import json
from pathlib import Path

from transformers import AutoTokenizer

from kurmanji_punctuation.dataset import (
    assert_no_article_overlap,
    build_processed_dataset,
    choose_largest_end_that_fits_xlmr,
    iter_word_windows,
    split_article_ids,
    tokens_and_labels_from_text,
    window_period_stats,
)


def test_extract_comma():
    tokens, labels = tokens_and_labels_from_text("Lê belê, em ê.")
    assert tokens == ["Lê", "belê", "em", "ê"]
    assert labels == ["O", "COMMA", "O", "PERIOD"]


def test_extract_period():
    tokens, labels = tokens_and_labels_from_text("Ez hatim.")
    assert tokens == ["Ez", "hatim"]
    assert labels == ["O", "PERIOD"]


def test_extract_question():
    tokens, labels = tokens_and_labels_from_text("Tu baş î?")
    assert tokens[-1] == "î"
    assert labels[-1] == "QUESTION"


def test_extract_exclamation():
    tokens, labels = tokens_and_labels_from_text("Were!")
    assert labels == ["EXCLAMATION"]


def _long_article(i: int) -> str:
    sent = f"Ev gotin hejmar {i} ye û mirov li Kurdistanê dijî. "
    return (sent * 40).strip()


def test_article_split_no_overlap(tmp_path: Path):
    raw = tmp_path / "raw.jsonl"
    rows = [{"article_id": f"wiki_{i:06d}", "text": _long_article(i)} for i in range(40)]
    raw.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    out = tmp_path / "processed"
    stats = build_processed_dataset(
        raw,
        out,
        seed=42,
        min_words_per_window=40,
        target_words_min=50,
        target_words_max=80,
        train_overlap_words=0,
        val_overlap_words=0,
        test_overlap_words=0,
    )
    assert stats["sample_mode"] == "continuous_word_windows"
    assert stats["article_overlap_check"] == "passed"
    assert stats["period_window_stats"]["train"]["fraction_of_period_labels_at_final_word"] < 0.5
    splits = {}
    for name in ("train", "validation", "test"):
        path = out / f"{name}.jsonl"
        splits[name] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    assert_no_article_overlap(splits)
    assert sum(len(v) for v in splits.values()) > 0


def test_split_article_ids_disjoint():
    ids = [f"a{i}" for i in range(100)]
    tr, va, te = split_article_ids(ids, seed=42)
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert len(tr) + len(va) + len(te) == 100


def test_windows_do_not_require_period_at_end():
    import random

    tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    text = ("Ez li Amedê dijîm. Navê min Azad e. Ew li Hewlêrê dixebite. " * 30).strip()
    tokens, labels = tokens_and_labels_from_text(text)
    assert tokens and labels
    samples = list(
        iter_word_windows(
            "wiki_x",
            tokens,
            labels,
            tokenizer=tok,
            rng=random.Random(0),
            max_model_length=256,
            min_words=40,
            target_words_min=50,
            target_words_max=90,
            overlap_words=0,
            absolute_min_words=20,
        )
    )
    assert len(samples) >= 2
    ends_period = sum(1 for s in samples if s["labels"][-1] == "PERIOD")
    assert ends_period < len(samples)  # not all windows end on PERIOD
    stats = window_period_stats(samples)
    assert stats["fraction_of_period_labels_at_final_word"] < 0.25


def test_choose_end_respects_max_length():
    tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    words = [f"peyva{i}" for i in range(300)]
    end = choose_largest_end_that_fits_xlmr(
        tok, words, 0, preferred_end=300, max_length=64
    )
    enc = tok(words[:end], is_split_into_words=True, add_special_tokens=True, truncation=False)
    assert len(enc["input_ids"]) <= 64
    assert end > 1
