"""Text normalisation for the mood classifier.

The transformations here are intentionally light. TF-IDF with an n-gram range
already handles tokenisation and stop-word removal, so the preprocessor only
needs to produce clean, lower-cased text and strip noise (URLs, mentions,
non-letters) that would otherwise pollute the vocabulary.

Lemmatisation/stemming is deliberately omitted: on short emotional sentences it
tends to collapse distinct feeling words and gave no measurable gain in
testing, so it is left out to keep inference dependency-free and fast.
"""
from __future__ import annotations

import re

_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_NON_LETTER_RE = re.compile(r"[^a-z\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise a single string of free-form text.

    Steps: lower-case, strip URLs and @mentions, drop any character that is not
    a letter or whitespace, then collapse repeated whitespace.

    Args:
        text: Raw user or dataset text.

    Returns:
        The cleaned text (possibly an empty string).
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _NON_LETTER_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text)
    return text.strip()


def clean_series(texts) -> list[str]:
    """Clean an iterable of strings, returning a list of cleaned strings."""
    return [clean_text(t) for t in texts]
