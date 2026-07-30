"""Text helpers for Kurmanji Latin filtering and word/label extraction."""

from __future__ import annotations

import re
import unicodedata

from .labels import PUNCT_CHARS, normalize_punct_char

# Arabic / Persian / Sorani blocks (reject if dominant).
ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
# Latin letters including Hawar diacritics (çêîşû and friends).
LATIN_LETTER_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]", re.UNICODE)
HAWAR_RE = re.compile(r"[ÇçÊêÎîŞşÛû]", re.UNICODE)

MULTISPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[\"'«]?[A-ZÇÊÎŞÛÁÉÍÓÚÄËÖÜ])")


def strip_wiki_markup(text: str) -> str:
    # Templates, HTML, bold/italic apostrophes, headings.
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"={2,}", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    return MULTISPACE_RE.sub(" ", text).strip()


def is_kurmanji_latin(text: str, min_latin_ratio: float = 0.85) -> bool:
    """Keep only Latin-script Kurmanji; drop Sorani/Arabic-heavy lines."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 12:
        return False
    arabic = sum(1 for c in letters if ARABIC_RE.match(c))
    if arabic / len(letters) > 0.05:
        return False
    latin = sum(1 for c in letters if LATIN_LETTER_RE.match(c))
    if latin / len(letters) < min_latin_ratio:
        return False
    # Prefer Hawar orthography when enough letters; allow short Latin without diacritics.
    if len(letters) >= 40 and not HAWAR_RE.search(text):
        # Still accept: many place names / short stubs lack diacritics.
        pass
    return True


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n{2,}|\n(?=[A-ZÇÊÎŞÛ])", text)
    return [MULTISPACE_RE.sub(" ", p).strip() for p in parts if p and p.strip()]


def split_sentences(paragraph: str) -> list[str]:
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    # Soft split; keep punctuation on the sentence.
    chunks = SENTENCE_SPLIT_RE.split(paragraph)
    out: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) >= 8:
            out.append(chunk)
    return out or ([paragraph] if len(paragraph) >= 8 else [])


_TOKEN_RE = re.compile(
    r"[0-9]+(?:[.,][0-9]+)*|"
    r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ſÇçÊêÎîŞşÛû]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿĀ-ſÇçÊêÎîŞşÛû]+)*|"
    r"[.?,:;\-!—…¿¡]|[^\s]",
    re.UNICODE,
)


def tokenize_with_punct(text: str) -> list[str]:
    """Whitespace-ish tokenizer that keeps punctuation as separate tokens."""
    text = unicodedata.normalize("NFC", text.strip())
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]


def words_and_labels(text: str, lowercase: bool = True) -> tuple[list[str], list[str]] | None:
    """
    Convert punctuated text into (words, labels) where each label is the
    punctuation that follows the word, or '0'. Matches FullStop / SEPP-NLG task 2.
    """
    tokens = tokenize_with_punct(text)
    if not tokens:
        return None

    words: list[str] = []
    labels: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in PUNCT_CHARS or normalize_punct_char(tok):
            # Leading / orphan punctuation — skip.
            i += 1
            continue
        if not any(c.isalnum() for c in tok):
            i += 1
            continue

        word = tok.lower() if lowercase else tok
        label = "0"
        j = i + 1
        while j < len(tokens):
            nxt = tokens[j]
            mapped = normalize_punct_char(nxt) if len(nxt) == 1 else None
            if mapped:
                label = mapped
                j += 1
                break
            if nxt in PUNCT_CHARS:
                j += 1
                continue
            break
        words.append(word)
        labels.append(label)
        i = j if j > i + 1 else i + 1

    if len(words) < 3:
        return None
    # Need at least some punctuation signal in the sentence for useful training.
    if all(l == "0" for l in labels):
        return None
    return words, labels


def strip_punctuation_for_asr_like(text: str) -> str:
    """Simulate omnilingual ASR output: lowercase, no punctuation."""
    tokens = tokenize_with_punct(text)
    words = []
    for tok in tokens:
        if normalize_punct_char(tok) or tok in PUNCT_CHARS:
            continue
        if any(c.isalnum() for c in tok):
            words.append(tok.lower())
    return " ".join(words)
