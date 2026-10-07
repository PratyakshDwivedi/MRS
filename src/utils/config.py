"""Central configuration: paths, constants and shared settings.

Keeping configuration separate from logic means paths and hyper-parameters
live in one place and the rest of the codebase imports from here.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
MOOD_MODEL_DIR: Path = MODELS_DIR / "mood_classifier"
MUSIC_MODEL_DIR: Path = MODELS_DIR / "music_clusterer"

ASSETS_DIR: Path = PROJECT_ROOT / "assets"

# Raw dataset files (populated by scripts/prepare_data.py)
EMOTION_TRAIN_CSV: Path = RAW_DATA_DIR / "emotion_train.csv"
EMOTION_VAL_CSV: Path = RAW_DATA_DIR / "emotion_validation.csv"
EMOTION_TEST_CSV: Path = RAW_DATA_DIR / "emotion_test.csv"
MUSIC_CSV: Path = RAW_DATA_DIR / "spotify_tracks.csv"

# Persisted model artifacts
MOOD_VECTORIZER_PATH: Path = MOOD_MODEL_DIR / "tfidf_vectorizer.joblib"
MOOD_CLASSIFIER_PATH: Path = MOOD_MODEL_DIR / "logistic_regression.joblib"
MOOD_METADATA_PATH: Path = MOOD_MODEL_DIR / "metadata.json"

MUSIC_SCALER_PATH: Path = MUSIC_MODEL_DIR / "scaler.joblib"
MUSIC_KMEANS_PATH: Path = MUSIC_MODEL_DIR / "kmeans.joblib"
MUSIC_TRACKS_PATH: Path = MUSIC_MODEL_DIR / "clustered_tracks.parquet"
MUSIC_METADATA_PATH: Path = MUSIC_MODEL_DIR / "metadata.json"

# --------------------------------------------------------------------------- #
# Emotion label mapping (dair-ai/emotion integer labels -> names)
# --------------------------------------------------------------------------- #
EMOTION_LABELS: dict[int, str] = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}
EMOTION_NAMES: list[str] = [EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS)]

# --------------------------------------------------------------------------- #
# Audio features used for clustering
# --------------------------------------------------------------------------- #
# These are the numeric Spotify audio features used to group tracks. They live
# on very different scales (tempo in BPM vs. valence in 0-1), which is why the
# clusterer standardises them before running K-Means.
CLUSTER_FEATURES: list[str] = [
    "danceability",
    "energy",
    "loudness",
    "acousticness",
    "valence",
    "tempo",
]

# Columns kept for display / recommendation output
TRACK_METADATA_COLS: list[str] = [
    "track_id",
    "track_name",
    "artists",
    "album_name",
    "track_genre",
    "popularity",
]

# Reproducibility
RANDOM_STATE: int = 42
