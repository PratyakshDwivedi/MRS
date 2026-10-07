"""Tests for the end-to-end recommendation engine."""
import pytest

from src.utils import config


def test_recommend_returns_requested_count(recommender):
    result = recommender.recommend("I am feeling low and tired today", n=5)
    assert result.predicted_mood in config.EMOTION_NAMES
    assert 1 <= len(result.recommendations) <= 5
    assert result.cluster in recommender.mood_to_cluster.values()


def test_recommendations_have_required_fields(recommender):
    result = recommender.recommend("I'm so happy and excited!", n=3)
    for rec in result.recommendations:
        assert rec.track_name
        assert rec.artists
        assert -1.0 <= rec.similarity <= 1.0


def test_recommendations_are_unique(recommender):
    result = recommender.recommend("I'm furious and angry", n=6)
    keys = {(r.track_name, r.artists) for r in result.recommendations}
    assert len(keys) == len(result.recommendations)


def test_empty_input_raises(recommender):
    with pytest.raises(ValueError):
        recommender.recommend("!!! 123 @@@", n=5)


def test_mapping_covers_all_profiled_moods(recommender):
    from src.recommendation.recommender import MOOD_PROFILES

    for mood in MOOD_PROFILES:
        assert mood in recommender.mood_to_cluster
