"""Tests for Unicode / Kurmanji normalization."""

from kurmanji_punctuation.normalization import normalize_for_dataset, to_nfc


def test_nfc_normalization():
    # e + combining circumflex → ê
    decomposed = "e\u0302"
    assert to_nfc(decomposed) == "ê"


def test_preserves_kurmanji_letters():
    text = "Çêşîû çêşîû"
    out = normalize_for_dataset(text, map_ellipsis_to_period=True)
    for ch in "çÇêÊîÎşŞûÛ":
        if ch in text:
            assert ch in out


def test_no_turkish_i_swap():
    text = "Iqlimî i"
    out = normalize_for_dataset(text)
    assert "Iqlimî" in out or "Iqlimî".lower() in out.lower()
    # Must not turn ASCII I into İ or i into ı
    assert "ı" not in out
    assert "İ" not in out


def test_ellipsis_to_period():
    text = "Ez hatim… Ew çû."
    out = normalize_for_dataset(text, map_ellipsis_to_period=True)
    assert "…" not in out
    assert "." in out
