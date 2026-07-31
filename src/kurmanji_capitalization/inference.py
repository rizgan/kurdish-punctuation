"""Sliding-window capitalization restoration with case-only preservation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .casing import apply_case_label, first_letter_index, kurmanji_lower
from .constants import ID2LABEL, LABEL2ID, LABELS, SUPPORTED_PUNCT
from .normalization import normalize_for_inference
from .sentence_rule import SentenceRuleConfig, capitalize_sentence_starts, sentence_start_word_indices
from .text_utils import (
    assert_case_only,
    is_email,
    is_numeric_token,
    is_url,
    reconstruct_with_spacing,
    tokenize_words_and_punct,
)


class CapitalizationRestorer:
    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        max_length: int = 256,
        overlap_words: int = 32,
        batch_size: int = 16,
        title_threshold: float = 0.80,
        upper_threshold: float = 0.90,
        sentence_cfg: SentenceRuleConfig | None = None,
        minimum_confidence: dict[str, float] | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_length = max_length
        self.overlap_words = overlap_words
        self.batch_size = batch_size
        self.sentence_cfg = sentence_cfg or SentenceRuleConfig()
        conf = minimum_confidence or {}
        self.title_threshold = float(conf.get("TITLE", title_threshold))
        self.upper_threshold = float(conf.get("UPPER", upper_threshold))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    def _windows(self, tokens: list[str]) -> list[tuple[int, int]]:
        n = len(tokens)
        if n == 0:
            return []
        windows: list[tuple[int, int]] = []
        start = 0
        while start < n:
            lo, hi = start + 1, n
            best = start + 1
            while lo <= hi:
                mid = (lo + hi) // 2
                enc = self.tokenizer(
                    tokens[start:mid],
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
            windows.append((start, end))
            if end >= n:
                break
            start = max(end - self.overlap_words, start + 1)
        return windows

    @torch.inference_mode()
    def _logits_for_window(self, tokens: list[str]) -> np.ndarray:
        """Return [n_tokens, num_labels] logits for *first* subtoken of each token."""
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        word_ids = enc.word_ids(batch_index=0)
        enc_t = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc_t)
        logits = out.logits[0].detach().float().cpu().numpy()
        n = len(tokens)
        word_logits = np.zeros((n, len(LABELS)), dtype=np.float32)
        seen: set[int] = set()
        for i, wid in enumerate(word_ids):
            if wid is None or wid in seen or wid >= n:
                continue
            seen.add(wid)
            word_logits[wid] = logits[i]
        return word_logits

    def _aggregate_logits(self, tokens: list[str]) -> np.ndarray:
        windows = self._windows(tokens)
        n = len(tokens)
        acc = np.zeros((n, len(LABELS)), dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)
        for start, end in windows:
            w_logits = self._logits_for_window(tokens[start:end])
            for i in range(end - start):
                acc[start + i] += w_logits[i]
                counts[start + i] += 1.0
        counts = np.maximum(counts, 1.0)
        return (acc / counts[:, None]).astype(np.float32)

    def predict_tokens(self, punctuated_text: str) -> list[dict[str, Any]]:
        text = normalize_for_inference(punctuated_text)
        if not text:
            return []

        # Lower then sentence rule — same format as training input.
        lowered = kurmanji_lower(text)
        after_rule = capitalize_sentence_starts(lowered, self.sentence_cfg)
        tokens = tokenize_words_and_punct(after_rule)
        starts = sentence_start_word_indices(tokens, self.sentence_cfg)
        logits = self._aggregate_logits(tokens) if tokens else np.zeros((0, len(LABELS)))

        results: list[dict[str, Any]] = []
        for i, tok in enumerate(tokens):
            protected = (
                tok in SUPPORTED_PUNCT
                or is_url(tok)
                or is_email(tok)
                or is_numeric_token(tok)
                or first_letter_index(tok) is None
            )
            sentence_start = i in starts
            if protected:
                results.append(
                    {
                        "token_before": tok,
                        "token_after_rule": tok,
                        "predicted_label": "KEEP",
                        "confidence": 1.0,
                        "token_after": tok,
                        "protected": True,
                        "sentence_start": sentence_start,
                    }
                )
                continue

            probs = _softmax(logits[i])
            pred_id = int(np.argmax(probs))
            label = ID2LABEL[pred_id]
            conf = float(probs[pred_id])

            # Sentence-start: keep rule capitalization unless UPPER with high confidence.
            if sentence_start:
                if label == "UPPER" and conf >= self.upper_threshold:
                    out_tok = apply_case_label(tok, "UPPER")
                    final_label = "UPPER"
                else:
                    out_tok = tok  # already title-cased by rule
                    final_label = "KEEP"
                    conf = float(probs[LABEL2ID["KEEP"]])
                results.append(
                    {
                        "token_before": tok,
                        "token_after_rule": tok,
                        "predicted_label": final_label,
                        "confidence": conf,
                        "token_after": out_tok,
                        "protected": False,
                        "sentence_start": True,
                    }
                )
                continue

            final_label = "KEEP"
            if label == "TITLE" and conf >= self.title_threshold:
                final_label = "TITLE"
            elif label == "UPPER" and conf >= self.upper_threshold:
                final_label = "UPPER"
            out_tok = apply_case_label(tok, final_label)
            results.append(
                {
                    "token_before": tok,
                    "token_after_rule": tok,
                    "predicted_label": final_label,
                    "confidence": conf if final_label != "KEEP" else float(probs[LABEL2ID["KEEP"]]),
                    "token_after": out_tok,
                    "protected": False,
                    "sentence_start": False,
                }
            )
        return results

    def restore(self, punctuated_text: str) -> str:
        text = normalize_for_inference(punctuated_text)
        if not text:
            return text
        lowered = kurmanji_lower(text)
        after_rule = capitalize_sentence_starts(lowered, self.sentence_cfg)
        preds = self.predict_tokens(text)
        new_tokens = [p["token_after"] for p in preds]
        out = reconstruct_with_spacing(after_rule, new_tokens)
        assert_case_only(after_rule, out)
        return out


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)
