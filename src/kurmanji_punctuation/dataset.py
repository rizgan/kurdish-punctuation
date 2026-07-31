"""Dataset building: continuous word windows + article-level splits."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from .constants import (
    LABEL2ID,
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
            while j < len(raw_toks) and raw_toks[j] in SUPPORTED_PUNCT:
                j += 1
        tokens.append(tok)
        labels.append(label)
        i = j
    if not tokens:
        return None
    return tokens, labels


def split_into_sentence_spans(text: str) -> list[str]:
    """Split on . ? ! keeping the terminator on the sentence (eval helpers only)."""
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


def subtoken_length(tokenizer, words: list[str]) -> int:
    enc = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=False,
    )
    return len(enc["input_ids"])


def choose_largest_end_that_fits_xlmr(
    tokenizer,
    tokens: list[str],
    start: int,
    *,
    preferred_end: int,
    max_length: int,
) -> int:
    """
    Largest end in (start, preferred_end] such that tokens[start:end]
    encode to <= max_length subtokens (incl. special tokens).
    Does NOT snap to sentence boundaries.
    """
    preferred_end = min(preferred_end, len(tokens))
    if preferred_end <= start:
        return start
    lo, hi = start + 1, preferred_end
    best = start
    while lo <= hi:
        mid = (lo + hi) // 2
        if subtoken_length(tokenizer, tokens[start:mid]) <= max_length:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best == start and start < len(tokens):
        # Single-word fallback even if it truncates later at train time.
        best = start + 1
    return best


def iter_word_windows(
    article_id: str,
    tokens: list[str],
    labels: list[str],
    *,
    tokenizer,
    rng: random.Random,
    max_model_length: int = 256,
    min_words: int = 80,
    target_words_min: int = 110,
    target_words_max: int = 180,
    overlap_words: int = 8,
    absolute_min_words: int = 40,
) -> Iterator[dict[str, Any]]:
    """Slice a full-article token sequence into continuous windows."""
    n = len(tokens)
    if n < absolute_min_words:
        return
    if n < min_words:
        # Keep short-but-usable articles as a single sample (no tiny tails).
        yield {
            "id": f"{article_id}_000001",
            "article_id": article_id,
            "tokens": tokens,
            "labels": labels,
        }
        return

    start = 0
    sample_idx = 0
    while start < n:
        remaining = n - start
        if remaining < absolute_min_words and sample_idx > 0:
            break

        target = rng.randint(target_words_min, target_words_max)
        if remaining < min_words and sample_idx > 0:
            # Expand last window backward instead of emitting a stub.
            start = max(0, n - max(min_words, min(target, remaining + overlap_words)))
            remaining = n - start

        preferred_end = min(start + target, n)
        end = choose_largest_end_that_fits_xlmr(
            tokenizer,
            tokens,
            start,
            preferred_end=preferred_end,
            max_length=max_model_length,
        )
        if end <= start:
            break

        # If this would leave a tiny remainder, absorb it into this window if it fits.
        rem_after = n - end
        if 0 < rem_after < absolute_min_words:
            end2 = choose_largest_end_that_fits_xlmr(
                tokenizer,
                tokens,
                start,
                preferred_end=n,
                max_length=max_model_length,
            )
            if end2 > end:
                end = end2

        sample_idx += 1
        yield {
            "id": f"{article_id}_{sample_idx:06d}",
            "article_id": article_id,
            "tokens": tokens[start:end],
            "labels": labels[start:end],
        }

        if end >= n:
            break
        next_start = max(end - overlap_words, start + 1)
        if next_start >= n:
            break
        # Avoid infinite loop on pathological overlap.
        if next_start <= start:
            next_start = end
        start = next_start


def window_period_stats(rows: list[dict]) -> dict[str, Any]:
    period_total = 0
    period_at_last = 0
    hist = Counter()
    for row in rows:
        labs = row["labels"]
        n_per = sum(1 for x in labs if x == "PERIOD")
        period_total += n_per
        if labs and labs[-1] == "PERIOD":
            period_at_last += 1
        if n_per == 0:
            hist["samples_without_period"] += 1
        elif n_per == 1:
            hist["samples_with_1_period"] += 1
        else:
            hist["samples_with_2_or_more_periods"] += 1
    frac = (period_at_last / period_total) if period_total else 0.0
    return {
        "period_total": period_total,
        "period_at_last_word": period_at_last,
        "period_inside_window": period_total - period_at_last,
        "fraction_of_period_labels_at_final_word": frac,
        **dict(hist),
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
    tokenizer_name: str = "FacebookAI/xlm-roberta-base",
    max_model_length: int = 256,
    min_words_per_window: int = 80,
    target_words_min: int = 110,
    target_words_max: int = 180,
    train_overlap_words: int = 8,
    val_overlap_words: int = 0,
    test_overlap_words: int = 0,
    max_articles: int | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    articles = load_raw_jsonl(input_jsonl)
    if max_articles is not None:
        rng0 = random.Random(seed)
        articles = list(articles)
        rng0.shuffle(articles)
        articles = articles[:max_articles]

    train_ids, val_ids, test_ids = split_article_ids(
        [a["article_id"] for a in articles], seed=seed
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    rng = random.Random(seed)

    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    overlap_by_split = {
        "train": train_overlap_words,
        "validation": val_overlap_words,
        "test": test_overlap_words,
    }

    for art in articles:
        aid = art["article_id"]
        if aid in train_ids:
            bucket = "train"
        elif aid in val_ids:
            bucket = "validation"
        else:
            bucket = "test"

        text = normalize_for_dataset(art["text"], map_ellipsis_to_period=map_ellipsis_to_period)
        pair = tokens_and_labels_from_text(text)
        if not pair:
            continue
        tokens, labels = pair
        for sample in iter_word_windows(
            aid,
            tokens,
            labels,
            tokenizer=tokenizer,
            rng=rng,
            max_model_length=max_model_length,
            min_words=min_words_per_window,
            target_words_min=target_words_min,
            target_words_max=target_words_max,
            overlap_words=int(overlap_by_split[bucket]),
        ):
            splits[bucket].append(sample)

    assert_no_article_overlap(splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", splits["train"])
    write_jsonl(output_dir / "validation.jsonl", splits["validation"])
    write_jsonl(output_dir / "test.jsonl", splits["test"])

    period_stats = {k: window_period_stats(v) for k, v in splits.items()}
    stats = {
        "sample_mode": "continuous_word_windows",
        "n_articles": len(articles),
        "n_articles_train": len(train_ids),
        "n_articles_validation": len(val_ids),
        "n_articles_test": len(test_ids),
        "n_samples": {k: len(v) for k, v in splits.items()},
        "label_counts": {k: label_distribution(v) for k, v in splits.items()},
        "period_window_stats": period_stats,
        "seed": seed,
        "tokenizer_name": tokenizer_name,
        "max_model_length": max_model_length,
        "min_words_per_window": min_words_per_window,
        "target_words_min": target_words_min,
        "target_words_max": target_words_max,
        "train_overlap_words": train_overlap_words,
        "val_overlap_words": val_overlap_words,
        "test_overlap_words": test_overlap_words,
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
