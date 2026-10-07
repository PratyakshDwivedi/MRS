"""Shared pytest fixtures.

Tests that need trained artifacts load them once and skip gracefully if the
models have not been trained yet, so the suite never fails on a fresh clone.
"""
import pytest

from src.models.mood_classifier import MoodClassifier
from src.models.music_clusterer import MusicClusterer


@pytest.fixture(scope="session")
def mood_model():
    try:
        return MoodClassifier.load()
    except FileNotFoundError:
        pytest.skip("mood model not trained; run scripts/train_mood.py")


@pytest.fixture(scope="session")
def clusterer():
    try:
        return MusicClusterer.load()
    except FileNotFoundError:
        pytest.skip("music model not trained; run scripts/train_clusters.py")


@pytest.fixture(scope="session")
def recommender():
    from src.recommendation import MusicRecommender

    try:
        return MusicRecommender.load()
    except FileNotFoundError:
        pytest.skip("models not trained; run the training scripts")
