"""Production pipeline: punctuation → sentence-start → capitalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from kurmanji_capitalization.casing import kurmanji_lower
from kurmanji_capitalization.inference import CapitalizationRestorer
from kurmanji_capitalization.sentence_rule import capitalize_sentence_starts
from kurmanji_capitalization.text_utils import validate_case_only_transformation
from kurmanji_punctuation.inference import PunctuationRestorer, TextPreservationError
from kurmanji_punctuation.normalization import normalize_for_inference
from kurmanji_punctuation.text_utils import validate_text_preservation

# Production defaults from capitalization threshold tuning (user target).
DEFAULT_TITLE_THRESHOLD = 0.80
DEFAULT_UPPER_THRESHOLD = 0.85


@dataclass(frozen=True)
class PipelineResult:
    input: str
    punctuated: str | None
    sentence_capitalized: str | None
    output: str
    preservation: dict[str, bool]
    mode: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Match the public JSON contract (omit mode unless useful for diagnostics).
        return {
            "input": d["input"],
            "punctuated": d["punctuated"],
            "sentence_capitalized": d["sentence_capitalized"],
            "output": d["output"],
            "preservation": d["preservation"],
            "mode": d["mode"],
        }


class TextRestorationPipeline:
    """Frozen punctuation v2 + capitalization v1 production stack."""

    def __init__(
        self,
        *,
        punctuation: PunctuationRestorer | None = None,
        capitalization: CapitalizationRestorer | None = None,
    ):
        self.punctuation = punctuation
        self.capitalization = capitalization

    def run(self, text: str, mode: str = "full") -> PipelineResult:
        if mode not in {"full", "punctuation", "capitalization"}:
            raise ValueError(f"Unknown mode: {mode!r}")

        raw = text if isinstance(text, str) else str(text)
        normalized = normalize_for_inference(raw)

        if mode == "punctuation":
            return self._run_punctuation(normalized)
        if mode == "capitalization":
            return self._run_capitalization(normalized)
        return self._run_full(normalized)

    def _run_punctuation(self, text: str) -> PipelineResult:
        if self.punctuation is None:
            raise RuntimeError("Punctuation model is required for mode=punctuation")
        if text == "":
            return PipelineResult(
                input="",
                punctuated="",
                sentence_capitalized=None,
                output="",
                preservation={"punctuation_only": True, "case_only": True},
                mode="punctuation",
            )
        punctuated = self.punctuation.restore(text)
        punct_ok = validate_text_preservation(text, punctuated)
        if not punct_ok:
            raise TextPreservationError(
                "Punctuation stage changed more than supported marks (, . ? !)"
            )
        return PipelineResult(
            input=text,
            punctuated=punctuated,
            sentence_capitalized=None,
            output=punctuated,
            preservation={"punctuation_only": punct_ok, "case_only": True},
            mode="punctuation",
        )

    def _run_capitalization(self, text: str) -> PipelineResult:
        if self.capitalization is None:
            raise RuntimeError("Capitalization model is required for mode=capitalization")
        if text == "":
            return PipelineResult(
                input="",
                punctuated=None,
                sentence_capitalized="",
                output="",
                preservation={"punctuation_only": True, "case_only": True},
                mode="capitalization",
            )
        sentence_capitalized = capitalize_sentence_starts(kurmanji_lower(text))
        output = self.capitalization.restore(text)
        case_ok = validate_case_only_transformation(sentence_capitalized, output)
        if not case_ok:
            raise RuntimeError("Capitalization stage changed more than letter case")
        return PipelineResult(
            input=text,
            punctuated=None,
            sentence_capitalized=sentence_capitalized,
            output=output,
            preservation={"punctuation_only": True, "case_only": case_ok},
            mode="capitalization",
        )

    def _run_full(self, text: str) -> PipelineResult:
        if self.punctuation is None or self.capitalization is None:
            raise RuntimeError("Both models are required for mode=full")
        if text == "":
            return PipelineResult(
                input="",
                punctuated="",
                sentence_capitalized="",
                output="",
                preservation={"punctuation_only": True, "case_only": True},
                mode="full",
            )

        # 1–2) punctuation + punctuation-only preservation
        punctuated = self.punctuation.restore(text)
        punct_ok = validate_text_preservation(text, punctuated)
        if not punct_ok:
            raise TextPreservationError(
                "Punctuation stage changed more than supported marks (, . ? !)"
            )

        # 3) sentence-start capitalization rule
        sentence_capitalized = capitalize_sentence_starts(kurmanji_lower(punctuated))

        # 4) capitalization model (internally re-applies lower + sentence rule)
        output = self.capitalization.restore(punctuated)
        case_ok = validate_case_only_transformation(sentence_capitalized, output)
        if not case_ok:
            raise RuntimeError("Capitalization stage changed more than letter case")

        return PipelineResult(
            input=text,
            punctuated=punctuated,
            sentence_capitalized=sentence_capitalized,
            output=output,
            preservation={"punctuation_only": punct_ok, "case_only": case_ok},
            mode="full",
        )
