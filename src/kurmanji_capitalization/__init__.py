"""Kurmanji Latin capitalization restoration (XLM-RoBERTa token classification)."""

__version__ = "0.1.0"

from .constants import LABELS, MACRO_F1_LABELS
from .inference import CapitalizationRestorer
from .sentence_rule import SentenceRuleConfig, capitalize_sentence_starts

__all__ = [
    "LABELS",
    "MACRO_F1_LABELS",
    "CapitalizationRestorer",
    "SentenceRuleConfig",
    "capitalize_sentence_starts",
    "__version__",
]
