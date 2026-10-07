"""Tests for text preprocessing."""
from src.preprocessing import clean_text, clean_series


def test_lowercases_and_strips_punctuation():
    assert clean_text("I AM so HAPPY!!!") == "i am so happy"


def test_removes_urls_and_mentions():
    out = clean_text("check http://x.co now @friend :)")
    assert "http" not in out and "@friend" not in out
    assert "check" in out and "now" in out


def test_collapses_whitespace():
    assert clean_text("too    many     spaces") == "too many spaces"


def test_handles_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("12345 !!!") == ""  # nothing but digits/punct


def test_clean_series_returns_list():
    out = clean_series(["Hello!", "WORLD"])
    assert out == ["hello", "world"]
