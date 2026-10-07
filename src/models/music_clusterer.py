"""Unsupervised music clustering: StandardScaler + K-Means.

Design notes
------------
* K-Means uses Euclidean distance, so features on different scales (tempo in
  the hundreds of BPM vs. valence in 0-1) must be standardised first, otherwise
  tempo dominates the distance. A ``StandardScaler`` is fitted on the training
  features and persisted; inference reuses it (never refits).
* K is chosen empirically by scanning a range of candidate values and
  inspecting inertia (elbow) and the silhouette score, rather than hard-coding.
* The fitted scaler, K-Means model and the per-track cluster assignments are
  all persisted so recommendation never needs to retrain.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils import config


@dataclass
class KSelectionResult:
    """Metrics for a single candidate value of K."""

    k: int
    inertia: float
    silhouette: float


class MusicClusterer:
    """StandardScaler + K-Means over Spotify audio features."""

    def __init__(
        self,
        scaler: StandardScaler | None = None,
        kmeans: KMeans | None = None,
        features: list[str] | None = None,
    ) -> None:
        self.scaler = scaler
        self.kmeans = kmeans
        self.features = features or config.CLUSTER_FEATURES

    # ------------------------------------------------------------------ #
    # K selection
    # ------------------------------------------------------------------ #
    def evaluate_k(
        self,
        df: pd.DataFrame,
        k_values: range | list[int],
        *,
        silhouette_sample: int = 10_000,
    ) -> list[KSelectionResult]:
        """Compute inertia and silhouette for each candidate K.

        The silhouette score is computed on a random sample for speed on large
        datasets; inertia uses the full data.
        """
        X = self._scale(df, fit=True)
        rng = np.random.default_rng(config.RANDOM_STATE)
        sample_idx = (
            rng.choice(X.shape[0], silhouette_sample, replace=False)
            if X.shape[0] > silhouette_sample
            else np.arange(X.shape[0])
        )
        results: list[KSelectionResult] = []
        for k in k_values:
            km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                        random_state=config.RANDOM_STATE)
            labels = km.fit_predict(X)
            sil = silhouette_score(X[sample_idx], labels[sample_idx])
            results.append(KSelectionResult(k=k, inertia=float(km.inertia_),
                                            silhouette=float(sil)))
        return results

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, n_clusters: int) -> "MusicClusterer":
        """Fit the scaler and K-Means with the chosen number of clusters."""
        X = self._scale(df, fit=True)
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=10,
            random_state=config.RANDOM_STATE,
        )
        self.kmeans.fit(X)
        return self

    def assign_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with a ``cluster`` column added."""
        self._ensure_loaded()
        X = self._scale(df, fit=False)
        out = df.copy()
        out["cluster"] = self.kmeans.predict(X)
        return out

    def centroids_original_scale(self) -> pd.DataFrame:
        """Cluster centroids expressed back in the original feature units."""
        self._ensure_loaded()
        centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        return pd.DataFrame(centers, columns=self.features).rename_axis("cluster")

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _scale(self, df: pd.DataFrame, *, fit: bool) -> np.ndarray:
        values = df[self.features].to_numpy(dtype=float)
        if fit:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(values)
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted/loaded.")
        return self.scaler.transform(values)

    def _ensure_loaded(self) -> None:
        if self.scaler is None or self.kmeans is None:
            raise RuntimeError(
                "MusicClusterer is not trained/loaded. Train it or call "
                "MusicClusterer.load() first."
            )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, metadata: dict | None = None) -> None:
        """Persist the scaler, K-Means model and optional metadata."""
        self._ensure_loaded()
        config.MUSIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, config.MUSIC_SCALER_PATH)
        joblib.dump(self.kmeans, config.MUSIC_KMEANS_PATH)
        if metadata is not None:
            config.MUSIC_METADATA_PATH.write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

    @classmethod
    def load(cls) -> "MusicClusterer":
        """Load a persisted clusterer from disk."""
        if not config.MUSIC_SCALER_PATH.exists() or not config.MUSIC_KMEANS_PATH.exists():
            raise FileNotFoundError(
                "Music model artifacts not found. Train first with "
                "`python scripts/train_clusters.py`."
            )
        features = config.CLUSTER_FEATURES
        if config.MUSIC_METADATA_PATH.exists():
            meta = json.loads(config.MUSIC_METADATA_PATH.read_text(encoding="utf-8"))
            features = meta.get("features", features)
        return cls(
            scaler=joblib.load(config.MUSIC_SCALER_PATH),
            kmeans=joblib.load(config.MUSIC_KMEANS_PATH),
            features=features,
        )
