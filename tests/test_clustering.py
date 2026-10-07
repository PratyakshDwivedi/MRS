"""Tests for the music clusterer."""
import numpy as np
import pandas as pd
import pytest

from src.models.music_clusterer import MusicClusterer
from src.utils import config


def _toy_df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    # Two separated blobs in feature space so clustering is meaningful.
    a = rng.normal(0.2, 0.05, size=(n // 2, len(config.CLUSTER_FEATURES)))
    b = rng.normal(0.8, 0.05, size=(n // 2, len(config.CLUSTER_FEATURES)))
    data = np.vstack([a, b])
    return pd.DataFrame(data, columns=config.CLUSTER_FEATURES)


def test_untrained_assign_raises():
    with pytest.raises(RuntimeError):
        MusicClusterer().assign_clusters(_toy_df())


def test_fit_and_assign_in_range():
    df = _toy_df()
    clusterer = MusicClusterer().fit(df, n_clusters=2)
    out = clusterer.assign_clusters(df)
    assert "cluster" in out.columns
    assert set(out["cluster"].unique()).issubset({0, 1})
    assert len(out) == len(df)


def test_evaluate_k_returns_metrics():
    df = _toy_df()
    results = MusicClusterer().evaluate_k(df, [2, 3], silhouette_sample=1000)
    assert len(results) == 2
    assert all(-1.0 <= r.silhouette <= 1.0 for r in results)
    assert all(r.inertia >= 0 for r in results)


def test_loaded_clusterer_predicts_valid_clusters(clusterer):
    n_clusters = clusterer.kmeans.n_clusters
    sample = pd.DataFrame(
        [{f: 0.5 for f in clusterer.features}, {f: 0.1 for f in clusterer.features}]
    )
    out = clusterer.assign_clusters(sample)
    assert out["cluster"].between(0, n_clusters - 1).all()
