"""Align word-level capitalization labels to first SentencePiece subtoken."""

from __future__ import annotations


def align_labels_to_first_subtoken(
    word_ids: list[int | None],
    word_labels: list[int],
) -> list[int]:
    """
    Place each word label on the *first* subtoken of that word.
    Remaining subtokens and special/padding tokens get -100.
    """
    n = len(word_ids)
    aligned = [-100] * n
    if n == 0:
        return aligned

    seen: set[int] = set()
    for i, wid in enumerate(word_ids):
        if wid is None:
            aligned[i] = -100
            continue
        if wid in seen:
            aligned[i] = -100
            continue
        seen.add(wid)
        if wid < 0 or wid >= len(word_labels):
            aligned[i] = -100
        else:
            aligned[i] = int(word_labels[wid])
    return aligned
