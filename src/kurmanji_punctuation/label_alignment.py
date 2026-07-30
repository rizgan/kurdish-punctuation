"""Align word-level punctuation labels to XLM-R subtokens (last subtoken only)."""

from __future__ import annotations


def align_labels_to_last_subtoken(
    word_ids: list[int | None],
    word_labels: list[int],
) -> list[int]:
    """
    Place each word label on the *last* subtoken of that word.
    All other subtokens and special/padding tokens get -100.
    """
    n = len(word_ids)
    aligned = [-100] * n
    if n == 0:
        return aligned

    for i, wid in enumerate(word_ids):
        if wid is None:
            aligned[i] = -100
            continue
        # Last subtoken of this word: next word_id differs or ends / hits None.
        next_wid = word_ids[i + 1] if i + 1 < n else None
        is_last = next_wid != wid
        if is_last:
            if wid < 0 or wid >= len(word_labels):
                aligned[i] = -100
            else:
                aligned[i] = int(word_labels[wid])
        else:
            aligned[i] = -100
    return aligned
