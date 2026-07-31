"""Capitalization dataset: label derivation + continuous windows + article splits."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from .casing import (
    apply_case_label,
    first_letter_index,
    is_all_lower_letters,
    is_all_upper_letters,
    is_title_case_letters,
    kurmanji_lower,
    to_nfc,
)
from .constants import IGNORE_LABEL, LABEL2ID, LABELS, SUPPORTED_PUNCT
from .normalization import normalize_for_dataset
from .sentence_rule import SentenceRuleConfig, capitalize_sentence_starts, sentence_start_word_indices
from .text_utils import is_email, is_numeric_token, is_url, tokenize_words_and_punct

WIKI_MARKUP_RE = re.compile(r"\[\[|\]\]|\{\{|\}\}|={2,}|__+|</?[a-zA-Z]+>")
# Kurmanji Latin + common Latin extensions only (drop Arabic script noise from kuwiki).
LATIN_LETTER_RE = re.compile(
    r"^[A-Za-zÀ-ÖØ-öø-ÿĀ-ſÇçÊêÎîŞşÛûİı0-9'’\-]+$"
)


def is_protected_token(token: str) -> bool:
    if token in SUPPORTED_PUNCT:
        return True
    if is_url(token) or is_email(token) or is_numeric_token(token):
        return True
    if WIKI_MARKUP_RE.search(token):
        return True
    return False


def is_kurmanji_latin_token(token: str) -> bool:
    return bool(LATIN_LETTER_RE.match(token))


def letter_count(token: str) -> int:
    from .casing import letter_chars

    return len(letter_chars(token))


def is_unsupported_mixed_case(token: str) -> bool:
    """True if casing cannot be expressed as KEEP / TITLE / UPPER from lower+rule input."""
    if first_letter_index(token) is None:
        return False
    if is_all_lower_letters(token) or is_title_case_letters(token) or is_all_upper_letters(token):
        return False
    # e.g. McDonald / iPhone style
    return True


def derive_gold_label(original_token: str) -> str:
    """Map original token casing to KEEP / TITLE / UPPER / IGNORE (pre-rule filters)."""
    if is_protected_token(original_token):
        return IGNORE_LABEL
    if first_letter_index(original_token) is None:
        return IGNORE_LABEL
    if not is_kurmanji_latin_token(original_token):
        return IGNORE_LABEL
    if is_unsupported_mixed_case(original_token):
        return IGNORE_LABEL
    n_letters = letter_count(original_token)
    if is_all_upper_letters(original_token):
        # Acronyms need >= 2 letters (already enforced in is_all_upper_letters).
        return "UPPER"
    if is_title_case_letters(original_token):
        # Single-letter "titles" are almost always wiki/editorial noise.
        if n_letters < 2:
            return IGNORE_LABEL
        return "TITLE"
    if is_all_lower_letters(original_token):
        return "KEEP"
    return IGNORE_LABEL


def tokens_and_labels_from_article(
    text: str,
    *,
    sentence_cfg: SentenceRuleConfig | None = None,
) -> tuple[list[str], list[str], list[str]] | None:
    """
    Returns (model_input_tokens, labels, original_tokens).

    Pipeline:
      original → lower → sentence rule → model input tokens
      labels compare original casing; sentence starts / punct → IGNORE
    """
    sentence_cfg = sentence_cfg or SentenceRuleConfig()
    text = to_nfc(text)
    if not text.strip():
        return None

    original_tokens = tokenize_words_and_punct(text)
    if not original_tokens:
        return None

    lowered = kurmanji_lower(text)
    # Preserve punctuation; only letter case changes.
    after_rule = capitalize_sentence_starts(lowered, sentence_cfg)
    input_tokens = tokenize_words_and_punct(after_rule)
    if len(input_tokens) != len(original_tokens):
        # Rare tokenizer edge case — skip article.
        return None

    starts = sentence_start_word_indices(input_tokens, sentence_cfg)
    labels: list[str] = []
    for i, (orig, inp) in enumerate(zip(original_tokens, input_tokens)):
        if i in starts:
            labels.append(IGNORE_LABEL)
            continue
        if is_protected_token(orig) or orig in SUPPORTED_PUNCT:
            labels.append(IGNORE_LABEL)
            continue
        gold = derive_gold_label(orig)
        if gold == IGNORE_LABEL:
            labels.append(IGNORE_LABEL)
            continue
        # Consistency: applying gold to model input should match original casing
        # (after ignoring sentence-start which we already skipped).
        predicted = apply_case_label(inp, gold)
        if kurmanji_lower(predicted) != kurmanji_lower(orig):
            labels.append(IGNORE_LABEL)
            continue
        if gold != "KEEP" and predicted != orig:
            # Still accept if case-fold equal and shape matches intended class.
            if gold == "TITLE" and is_title_case_letters(orig):
                labels.append("TITLE")
            elif gold == "UPPER" and is_all_upper_letters(orig):
                labels.append("UPPER")
            else:
                labels.append(IGNORE_LABEL)
            continue
        labels.append(gold)

    return input_tokens, labels, original_tokens


def subtoken_length(tokenizer, words: list[str]) -> int:
    enc = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=False,
    )
    return len(enc["input_ids"])


def choose_largest_end_that_fits(
    tokenizer,
    tokens: list[str],
    start: int,
    *,
    preferred_end: int,
    max_length: int,
) -> int:
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
        best = start + 1
    return best


def iter_token_windows(
    article_id: str,
    tokens: list[str],
    labels: list[str],
    *,
    tokenizer,
    rng: random.Random,
    max_model_length: int = 256,
    min_tokens: int = 80,
    target_tokens_min: int = 110,
    target_tokens_max: int = 180,
    overlap_tokens: int = 8,
    absolute_min_tokens: int = 40,
) -> Iterator[dict[str, Any]]:
    n = len(tokens)
    if n < absolute_min_tokens:
        return
    if n < min_tokens:
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
        if remaining < absolute_min_tokens and sample_idx > 0:
            break

        target = rng.randint(target_tokens_min, target_tokens_max)
        if remaining < min_tokens and sample_idx > 0:
            start = max(0, n - max(min_tokens, min(target, remaining + overlap_tokens)))
            remaining = n - start

        preferred_end = min(start + target, n)
        end = choose_largest_end_that_fits(
            tokenizer,
            tokens,
            start,
            preferred_end=preferred_end,
            max_length=max_model_length,
        )
        if end <= start:
            break

        rem_after = n - end
        if 0 < rem_after < absolute_min_tokens:
            end2 = choose_largest_end_that_fits(
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
        next_start = max(end - overlap_tokens, start + 1)
        if next_start >= n:
            break
        if next_start <= start:
            next_start = end
        start = next_start


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
    out = {lab: int(c.get(lab, 0)) for lab in LABELS}
    out[IGNORE_LABEL] = int(c.get(IGNORE_LABEL, 0))
    return out


def count_words(tokens: list[str]) -> int:
    return sum(1 for t in tokens if t not in SUPPORTED_PUNCT and first_letter_index(t) is not None)


def build_processed_dataset(
    input_jsonl: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    map_ellipsis_to_period: bool = True,
    tokenizer_name: str = "FacebookAI/xlm-roberta-base",
    max_model_length: int = 256,
    min_tokens_per_window: int = 80,
    target_tokens_min: int = 110,
    target_tokens_max: int = 180,
    train_overlap_tokens: int = 8,
    val_overlap_tokens: int = 0,
    test_overlap_tokens: int = 0,
    max_articles: int | None = None,
    max_train_words: int | None = None,
    sentence_cfg: SentenceRuleConfig | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    sentence_cfg = sentence_cfg or SentenceRuleConfig()
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
        "train": train_overlap_tokens,
        "validation": val_overlap_tokens,
        "test": test_overlap_tokens,
    }
    skipped_token_mismatch = 0
    word_counts = {"train": 0, "validation": 0, "test": 0}

    for art in articles:
        aid = art["article_id"]
        if aid in train_ids:
            bucket = "train"
        elif aid in val_ids:
            bucket = "validation"
        else:
            bucket = "test"

        if max_train_words is not None and bucket == "train" and word_counts["train"] >= max_train_words:
            continue

        text = normalize_for_dataset(art["text"], map_ellipsis_to_period=map_ellipsis_to_period)
        triple = tokens_and_labels_from_article(text, sentence_cfg=sentence_cfg)
        if not triple:
            skipped_token_mismatch += 1
            continue
        tokens, labels, _orig = triple
        art_words = count_words(tokens)

        for sample in iter_token_windows(
            aid,
            tokens,
            labels,
            tokenizer=tokenizer,
            rng=rng,
            max_model_length=max_model_length,
            min_tokens=min_tokens_per_window,
            target_tokens_min=target_tokens_min,
            target_tokens_max=target_tokens_max,
            overlap_tokens=int(overlap_by_split[bucket]),
        ):
            splits[bucket].append(sample)
        word_counts[bucket] += art_words

    assert_no_article_overlap(splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", splits["train"])
    write_jsonl(output_dir / "validation.jsonl", splits["validation"])
    write_jsonl(output_dir / "test.jsonl", splits["test"])

    stats = {
        "sample_mode": "continuous_token_windows",
        "n_articles": len(articles),
        "n_articles_train": len(train_ids),
        "n_articles_validation": len(val_ids),
        "n_articles_test": len(test_ids),
        "n_samples": {k: len(v) for k, v in splits.items()},
        "word_counts_approx": word_counts,
        "label_counts": {k: label_distribution(v) for k, v in splits.items()},
        "skipped_token_mismatch_or_empty": skipped_token_mismatch,
        "seed": seed,
        "tokenizer_name": tokenizer_name,
        "max_model_length": max_model_length,
        "min_tokens_per_window": min_tokens_per_window,
        "target_tokens_min": target_tokens_min,
        "target_tokens_max": target_tokens_max,
        "train_overlap_tokens": train_overlap_tokens,
        "max_train_words": max_train_words,
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
    out: list[int] = []
    for x in labels:
        if x == IGNORE_LABEL or x == -100:
            out.append(-100)
        else:
            out.append(LABEL2ID[x])
    return out
