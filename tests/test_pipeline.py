"""Unit tests for production pipeline staging (no model weights required)."""

from __future__ import annotations

from kurmanji_pipeline.pipeline import PipelineResult, TextRestorationPipeline
from kurmanji_punctuation.inference import TextPreservationError


class FakePunct:
    def __init__(self, out: str):
        self.out = out

    def restore(self, text: str) -> str:
        return self.out


class FakeCap:
    def __init__(self, out: str):
        self.out = out

    def restore(self, text: str) -> str:
        return self.out


def test_full_pipeline_json_shape():
    punct = FakePunct("ez li amedê dijîm. navê min azad e.")
    cap = FakeCap("Ez li Amedê dijîm. Navê min Azad e.")
    pipe = TextRestorationPipeline(punctuation=punct, capitalization=cap)
    result = pipe.run("ez li amedê dijîm navê min azad e", mode="full")
    assert isinstance(result, PipelineResult)
    d = result.to_dict()
    assert d["punctuated"] == "ez li amedê dijîm. navê min azad e."
    assert d["sentence_capitalized"].startswith("Ez ")
    assert "Navê" in d["sentence_capitalized"]
    assert d["output"] == "Ez li Amedê dijîm. Navê min Azad e."
    assert d["preservation"]["punctuation_only"] is True
    assert d["preservation"]["case_only"] is True


def test_empty_full():
    pipe = TextRestorationPipeline(punctuation=FakePunct(""), capitalization=FakeCap(""))
    r = pipe.run("", mode="full")
    assert r.output == ""
    assert r.preservation == {"punctuation_only": True, "case_only": True}


def test_punctuation_mode():
    pipe = TextRestorationPipeline(
        punctuation=FakePunct("a b."),
        capitalization=None,
    )
    r = pipe.run("a b", mode="punctuation")
    assert r.output == "a b."
    assert r.punctuated == "a b."
    assert r.sentence_capitalized is None


def test_punctuation_preservation_failure():
    pipe = TextRestorationPipeline(
        punctuation=FakePunct("completely different words."),
        capitalization=FakeCap("x"),
    )
    try:
        pipe.run("ez li amedê", mode="full")
        assert False, "expected TextPreservationError"
    except TextPreservationError:
        pass
