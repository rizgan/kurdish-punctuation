"""Sliding-window punctuation restoration with text preservation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .constants import ID2LABEL, LABEL_TO_PUNCTUATION, LABELS, SUPPORTED_PUNCT
from .label_alignment import align_labels_to_last_subtoken
from .normalization import normalize_for_inference
from .text_utils import (
    extract_words,
    join_words_with_punctuation,
    tokenize_words_and_punct,
    validate_text_preservation,
)


class TextPreservationError(RuntimeError):
    pass


class PunctuationRestorer:
    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        max_length: int = 256,
        overlap_words: int = 32,
        batch_size: int = 16,
        minimum_confidence: dict[str, float] | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_length = max_length
        self.overlap_words = overlap_words
        self.batch_size = batch_size
        self.minimum_confidence = minimum_confidence or {
            "COMMA": 0.55,
            "PERIOD": 0.55,
            "QUESTION": 0.60,
            "EXCLAMATION": 0.70,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    def _windows(self, words: list[str]) -> list[tuple[int, int]]:
        """Return (start, end) word spans covering the full sequence with overlap."""
        n = len(words)
        if n == 0:
            return []
        # Estimate max words that fit in max_length (leave room for special tokens).
        # Empirically XLM-R may use >1 subtoken/word; probe with binary search per window.
        windows: list[tuple[int, int]] = []
        start = 0
        while start < n:
            lo, hi = start + 1, n
            best = start + 1
            while lo <= hi:
                mid = (lo + hi) // 2
                enc = self.tokenizer(
                    words[start:mid],
                    is_split_into_words=True,
                    truncation=False,
                    add_special_tokens=True,
                )
                if len(enc["input_ids"]) <= self.max_length:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            end = max(best, start + 1)
            # If even one word overflows, force single-word truncated encode later.
            windows.append((start, end))
            if end >= n:
                break
            start = max(end - self.overlap_words, start + 1)
        return windows

    @torch.inference_mode()
    def _logits_for_window(self, words: list[str]) -> np.ndarray:
        """Return [n_words, num_labels] logits for last-subtoken of each word."""
        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        word_ids = enc.word_ids(batch_index=0)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        logits = out.logits[0].detach().float().cpu().numpy()  # [T, C]
        n_words = len(words)
        word_logits = np.zeros((n_words, logits.shape[-1]), dtype=np.float64)
        counts = np.zeros(n_words, dtype=np.float64)
        # Collect last subtoken index per word_id
        last_idx: dict[int, int] = {}
        for i, wid in enumerate(word_ids):
            if wid is not None:
                last_idx[wid] = i
        for wid, ti in last_idx.items():
            if 0 <= wid < n_words:
                word_logits[wid] = logits[ti]
                counts[wid] = 1.0
        # Any missing word (truncated away): zeros — caller should avoid via windows.
        return word_logits

    def predict_tokens(self, text: str) -> list[dict[str, Any]]:
        text = normalize_for_inference(text)
        if text == "":
            return []

        # Preserve existing supported punctuation on input words.
        raw = tokenize_words_and_punct(text)
        words: list[str] = []
        existing_punct: list[str] = []
        i = 0
        while i < len(raw):
            tok = raw[i]
            if tok in SUPPORTED_PUNCT:
                i += 1
                continue
            punct = ""
            j = i + 1
            if j < len(raw) and raw[j] in SUPPORTED_PUNCT:
                punct = raw[j]
                j += 1
                while j < len(raw) and raw[j] in SUPPORTED_PUNCT:
                    j += 1
            words.append(tok)
            existing_punct.append(punct)
            i = j

        if not words:
            return []

        n = len(words)
        sum_logits = np.zeros((n, len(LABELS)), dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)

        for start, end in self._windows(words):
            window_words = words[start:end]
            w_logits = self._logits_for_window(window_words)
            for local_i in range(len(window_words)):
                g = start + local_i
                sum_logits[g] += w_logits[local_i]
                counts[g] += 1.0

        # Ensure no word was skipped.
        if np.any(counts == 0):
            missing = int(np.where(counts == 0)[0][0])
            raise RuntimeError(f"Word index {missing} was not covered by any window")

        mean_logits = sum_logits / counts[:, None]
        results: list[dict[str, Any]] = []
        for i, word in enumerate(words):
            if existing_punct[i]:
                # Do not duplicate: keep existing mark, skip model punct.
                lab = next(
                    (k for k, v in LABEL_TO_PUNCTUATION.items() if v == existing_punct[i]),
                    "O",
                )
                results.append(
                    {
                        "token": word,
                        "label": lab,
                        "punctuation": existing_punct[i],
                        "confidence": 1.0,
                    }
                )
                continue

            logits = mean_logits[i]
            # softmax
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
            pred_id = int(probs.argmax())
            label = ID2LABEL[pred_id]
            conf = float(probs[pred_id])
            if label != "O":
                thr = float(self.minimum_confidence.get(label, 0.0))
                if conf < thr:
                    label = "O"
                    conf = float(probs[LABELS.index("O")])
            results.append(
                {
                    "token": word,
                    "label": label,
                    "punctuation": LABEL_TO_PUNCTUATION[label],
                    "confidence": conf,
                }
            )
        return results

    def restore(self, text: str) -> str:
        text_n = normalize_for_inference(text)
        if text_n == "":
            return ""
        preds = self.predict_tokens(text_n)
        if not preds:
            return ""
        tokens = [p["token"] for p in preds]
        punct = [p["punctuation"] for p in preds]
        out = join_words_with_punctuation(tokens, punct)
        if not validate_text_preservation(text_n, out):
            raise TextPreservationError(
                "Restored text does not preserve input word/character sequence "
                f"(after stripping , . ? !). input={text_n!r} output={out!r}"
            )
        return out
