"""Recommendation engine: mood -> cluster -> ranked tracks.

How the mapping works
---------------------
Each emotion is described by a *preference vector* over the audio features
(e.g. "joy" prefers high valence / energy / danceability; "sadness" prefers
low valence / energy and higher acousticness). These preferences are expressed
as signed weights over the SAME features the clusterer uses.

The mood -> cluster mapping is then **derived from the data**: every cluster
centroid is standardised (z-scored against the corpus), scored against each
mood's preference vector, and each mood is attached to the cluster whose
centroid best matches its preferences. Nothing is hard-coded to a fixed cluster
id -- the mapping is recomputed from whatever centroids K-Means produced.

Within the selected cluster, candidate tracks are ranked by cosine similarity
between their standardised feature vector and the mood preference vector, so the
tracks that most strongly express the target mood surface first.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models.mood_classifier import MoodClassifier
from src.models.music_clusterer import MusicClusterer
from src.utils import config

# Signed preference weights per emotion over CLUSTER_FEATURES.
# +1 => the mood favours a high value of that feature, -1 => a low value.
# Order of keys is irrelevant; missing features default to 0.
MOOD_PROFILES: dict[str, dict[str, float]] = {
    "joy": {"valence": 1.0, "energy": 0.8, "danceability": 0.8, "acousticness": -0.3},
    "love": {"valence": 0.8, "energy": 0.2, "danceability": 0.4, "acousticness": 0.3},
    "sadness": {"valence": -1.0, "energy": -0.7, "acousticness": 0.7, "danceability": -0.3},
    "anger": {"valence": -0.6, "energy": 1.0, "loudness": 0.8, "acousticness": -0.6},
    "fear": {"valence": -0.7, "energy": -0.2, "acousticness": 0.4, "danceability": -0.2},
    "surprise": {"valence": 0.4, "energy": 0.7, "danceability": 0.5, "tempo": 0.4},
}


@dataclass
class Recommendation:
    """A single recommended track."""

    track_name: str
    artists: str
    album_name: str | None
    genre: str | None
    cluster: int
    similarity: float
    popularity: int | None
    spotify_url: str | None


@dataclass
class RecommendationResult:
    """Full result returned to the caller / UI."""

    predicted_mood: str
    confidence: float
    cluster: int
    recommendations: list[Recommendation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "predicted_mood": self.predicted_mood,
            "confidence": self.confidence,
            "cluster": self.cluster,
            "recommendations": [r.__dict__ for r in self.recommendations],
        }


class MusicRecommender:
    """Ties the mood classifier and music clusterer into one pipeline."""

    def __init__(
        self,
        mood_classifier: MoodClassifier,
        clusterer: MusicClusterer,
        tracks: pd.DataFrame,
    ) -> None:
        self.mood_classifier = mood_classifier
        self.clusterer = clusterer
        self.tracks = tracks
        self.features = clusterer.features
        # Pre-scale every track once for fast cosine ranking at request time.
        self._scaled = clusterer.scaler.transform(
            tracks[self.features].to_numpy(dtype=float)
        )
        self.mood_to_cluster = self._build_mapping()

    # ------------------------------------------------------------------ #
    # Mapping
    # ------------------------------------------------------------------ #
    def _profile_vector(self, mood: str) -> np.ndarray:
        prefs = MOOD_PROFILES.get(mood, {})
        return np.array([prefs.get(f, 0.0) for f in self.features], dtype=float)

    def _build_mapping(self) -> dict[str, int]:
        """Attach each mood to the best-matching cluster from the centroids."""
        # Standardise centroids against the corpus so "high"/"low" is relative.
        centers = self.clusterer.kmeans.cluster_centers_  # already in scaled space
        mapping: dict[str, int] = {}
        for mood in MOOD_PROFILES:
            target = self._profile_vector(mood)
            scores = centers @ target  # dot product in standardised space
            mapping[mood] = int(np.argmax(scores))
        return mapping

    def mapping_report(self) -> pd.DataFrame:
        """Human-readable mood -> cluster assignment table."""
        rows = [{"mood": m, "cluster": c} for m, c in self.mood_to_cluster.items()]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Recommendation
    # ------------------------------------------------------------------ #
    def recommend(self, text: str, n: int = 5) -> RecommendationResult:
        """Full pipeline: text -> mood -> cluster -> ranked tracks."""
        prediction = self.mood_classifier.predict(text)
        mood = prediction.mood
        # Fall back to a neutral-ish mood if a class has no profile.
        cluster = self.mood_to_cluster.get(mood, self.mood_to_cluster["joy"])

        mask = self.tracks["cluster"].to_numpy() == cluster
        candidate_idx = np.flatnonzero(mask)

        target = self._profile_vector(mood)
        norm = np.linalg.norm(target)
        if norm == 0:
            norm = 1.0
        target_unit = target / norm

        cand_vectors = self._scaled[candidate_idx]
        cand_norms = np.linalg.norm(cand_vectors, axis=1)
        cand_norms[cand_norms == 0] = 1.0
        sims = (cand_vectors @ target_unit) / cand_norms

        order = np.argsort(-sims)[: n * 4]  # shortlist, then dedupe by title
        recs: list[Recommendation] = []
        seen: set[str] = set()
        for pos in order:
            row = self.tracks.iloc[candidate_idx[pos]]
            key = f"{row.get('track_name')}|{row.get('artists')}".lower()
            if key in seen:
                continue
            seen.add(key)
            recs.append(self._row_to_rec(row, cluster, float(sims[pos])))
            if len(recs) >= n:
                break

        return RecommendationResult(
            predicted_mood=mood,
            confidence=prediction.confidence,
            cluster=cluster,
            recommendations=recs,
        )

    @staticmethod
    def _row_to_rec(row: pd.Series, cluster: int, similarity: float) -> Recommendation:
        track_id = row.get("track_id")
        url = f"https://open.spotify.com/track/{track_id}" if isinstance(track_id, str) else None
        pop = row.get("popularity")
        return Recommendation(
            track_name=str(row.get("track_name", "Unknown")),
            artists=str(row.get("artists", "Unknown")),
            album_name=(str(row["album_name"]) if "album_name" in row and pd.notna(row["album_name"]) else None),
            genre=(str(row["track_genre"]) if "track_genre" in row and pd.notna(row["track_genre"]) else None),
            cluster=cluster,
            similarity=round(similarity, 4),
            popularity=(int(pop) if pop is not None and pd.notna(pop) else None),
            spotify_url=url,
        )

    # ------------------------------------------------------------------ #
    # Convenience loader
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls) -> "MusicRecommender":
        """Load every persisted artifact and assemble the recommender."""
        mood = MoodClassifier.load()
        clusterer = MusicClusterer.load()
        if not config.MUSIC_TRACKS_PATH.exists():
            raise FileNotFoundError(
                "Clustered tracks not found. Train first with "
                "`python scripts/train_clusters.py`."
            )
        tracks = pd.read_parquet(config.MUSIC_TRACKS_PATH)
        return cls(mood, clusterer, tracks)
