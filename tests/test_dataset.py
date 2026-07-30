"""Tests for dataset labeling and article splits."""

import json
from pathlib import Path

from kurmanji_punctuation.dataset import (
    assert_no_article_overlap,
    build_processed_dataset,
    split_article_ids,
    tokens_and_labels_from_text,
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


def test_article_split_no_overlap(tmp_path: Path):
    raw = tmp_path / "raw.jsonl"
    rows = []
    for i in range(40):
        rows.append(
            {
                "article_id": f"wiki_{i:06d}",
                "text": f"Ev gotin hejmar {i} ye. Ev jî hevokeke din e.",
            }
        )
    raw.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    out = tmp_path / "processed"
    stats = build_processed_dataset(raw, out, seed=42, min_words=3, max_words=180)
    assert stats["article_overlap_check"] == "passed"
    # Load and re-check
    splits = {}
    for name in ("train", "validation", "test"):
        path = out / f"{name}.jsonl"
        splits[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert_no_article_overlap(splits)


def test_split_article_ids_disjoint():
    ids = [f"a{i}" for i in range(100)]
    tr, va, te = split_article_ids(ids, seed=42)
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert len(tr) + len(va) + len(te) == 100
