"""Tests for the mood classifier."""
import pytest

from src.models.mood_classifier import MoodClassifier
from src.utils import config


def test_untrained_predict_raises():
    with pytest.raises(RuntimeError):
        MoodClassifier().predict("hello")


def test_fit_predict_tiny():
    texts = [
        "i feel so happy and joyful today",
        "what a wonderful cheerful morning",
        "i am devastated and crying all night",
        "everything feels hopeless and sad",
    ]
    labels = ["joy", "joy", "sadness", "sadness"]
    clf = MoodClassifier().fit(texts, labels, min_df=1, ngram_range=(1, 1))
    pred = clf.predict("i am feeling cheerful and glad")
    assert pred.mood in {"joy", "sadness"}
    assert 0.0 <= pred.confidence <= 1.0
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-6


def test_empty_text_raises_after_cleaning():
    clf = MoodClassifier().fit(
        ["happy day", "sad night"], ["joy", "sadness"], min_df=1
    )
    with pytest.raises(ValueError):
        clf.predict("!!! 12345 @@@")


def test_loaded_model_predicts_valid_class(mood_model):
    pred = mood_model.predict("I am feeling low and tired today")
    assert pred.mood in config.EMOTION_NAMES
    assert pred.confidence > 0
