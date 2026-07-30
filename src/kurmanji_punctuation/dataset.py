"""Dataset building: sentences → tokens/labels; article-level splits."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from .constants import (
    LABEL2ID,
    LABEL_TO_PUNCTUATION,
    LABELS,
    PUNCTUATION_TO_LABEL,
    SENTENCE_END_PUNCT,
    SUPPORTED_PUNCT,
)
from .normalization import normalize_for_dataset
from .text_utils import tokenize_words_and_punct


def tokens_and_labels_from_text(text: str) -> tuple[list[str], list[str]] | None:
    """
    Convert punctuated text into (tokens, labels).
    Label is the punctuation that follows the word (COMMA/PERIOD/...).
    """
    raw_toks = tokenize_words_and_punct(text)
    tokens: list[str] = []
    labels: list[str] = []
    i = 0
    while i < len(raw_toks):
        tok = raw_toks[i]
        if tok in SUPPORTED_PUNCT:
            i += 1
            continue
        label = "O"
        j = i + 1
        if j < len(raw_toks) and raw_toks[j] in SUPPORTED_PUNCT:
            label = PUNCTUATION_TO_LABEL[raw_toks[j]]
            j += 1
            # Collapse doubled punctuation: keep first mapped label, skip extras.
            while j < len(raw_toks) and raw_toks[j] in SUPPORTED_PUNCT:
                j += 1
        tokens.append(tok)
        labels.append(label)
        i = j
    if not tokens:
        return None
    return tokens, labels


def split_into_sentence_spans(text: str) -> list[str]:
    """Split on . ? ! keeping the terminator on the sentence."""
    text = text.strip()
    if not text:
        return []
    parts: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in SENTENCE_END_PUNCT:
            sent = "".join(buf).strip()
            if sent:
                parts.append(sent)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def chunk_long_sample(
    tokens: list[str],
    labels: list[str],
    max_words: int,
) -> list[tuple[list[str], list[str]]]:
    """Split long sequences on COMMA boundaries when possible."""
    if len(tokens) <= max_words:
        return [(tokens, labels)]
    chunks: list[tuple[list[str], list[str]]] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + max_words, n)
        if end < n:
            # Prefer to break after a COMMA within the window.
            cut = None
            for k in range(end - 1, start + max(1, max_words // 3) - 1, -1):
                if labels[k] == "COMMA":
                    cut = k + 1
                    break
            if cut is not None:
                end = cut
        chunks.append((tokens[start:end], labels[start:end]))
        start = end
    return chunks


def iter_samples_from_article(
    article_id: str,
    text: str,
    *,
    map_ellipsis_to_period: bool = True,
    min_words: int = 3,
    max_words: int = 180,
) -> Iterator[dict[str, Any]]:
    text = normalize_for_dataset(text, map_ellipsis_to_period=map_ellipsis_to_period)
    if not text:
        return
    sample_idx = 0
    for sent in split_into_sentence_spans(text):
        pair = tokens_and_labels_from_text(sent)
        if not pair:
            continue
        tokens, labels = pair
        for tok_chunk, lab_chunk in chunk_long_sample(tokens, labels, max_words):
            if len(tok_chunk) < min_words:
                continue
            sample_idx += 1
            yield {
                "id": f"{article_id}_{sample_idx:06d}",
                "article_id": article_id,
                "tokens": tok_chunk,
                "labels": lab_chunk,
            }


def load_raw_jsonl(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({"article_id": str(obj["article_id"]), "text": str(obj["text"])})
    return rows


def split_article_ids(
    article_ids: list[str],
    *,
    seed: int = 42,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
) -> tuple[set[str], set[str], set[str]]:
    ids = sorted(set(article_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    test_ids = set(ids[n_train + n_val :])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    return train_ids, val_ids, test_ids


def assert_no_article_overlap(splits: dict[str, list[dict]]) -> None:
    sets = {name: {r["article_id"] for r in rows} for name, rows in splits.items()}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = sets[names[i]] & sets[names[j]]
            if inter:
                raise AssertionError(
                    f"article_id overlap between {names[i]} and {names[j]}: {list(inter)[:5]}"
                )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_distribution(rows: list[dict]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for row in rows:
        c.update(row["labels"])
    return {lab: int(c.get(lab, 0)) for lab in LABELS}


def build_processed_dataset(
    input_jsonl: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    map_ellipsis_to_period: bool = True,
    min_words: int = 3,
    max_words: int = 180,
) -> dict[str, Any]:
    articles = load_raw_jsonl(input_jsonl)
    train_ids, val_ids, test_ids = split_article_ids(
        [a["article_id"] for a in articles], seed=seed
    )
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for art in articles:
        aid = art["article_id"]
        if aid in train_ids:
            bucket = "train"
        elif aid in val_ids:
            bucket = "validation"
        else:
            bucket = "test"
        for sample in iter_samples_from_article(
            aid,
            art["text"],
            map_ellipsis_to_period=map_ellipsis_to_period,
            min_words=min_words,
            max_words=max_words,
        ):
            splits[bucket].append(sample)

    assert_no_article_overlap(splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", splits["train"])
    write_jsonl(output_dir / "validation.jsonl", splits["validation"])
    write_jsonl(output_dir / "test.jsonl", splits["test"])

    stats = {
        "n_articles": len(articles),
        "n_articles_train": len(train_ids),
        "n_articles_validation": len(val_ids),
        "n_articles_test": len(test_ids),
        "n_samples": {k: len(v) for k, v in splits.items()},
        "label_counts": {k: label_distribution(v) for k, v in splits.items()},
        "seed": seed,
        "article_overlap_check": "passed",
    }
    (output_dir / "statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def load_processed_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_label_ids(labels: list[str]) -> list[int]:
    return [LABEL2ID[lab] for lab in labels]


def decode_label_ids(ids: list[int]) -> list[str]:
    from .constants import ID2LABEL

    return [ID2LABEL[i] for i in ids]
