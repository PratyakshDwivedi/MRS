"""Download the raw datasets used by the Music Recommendation System.

Two public datasets are used:

1. Emotion text dataset (dair-ai/emotion) -- labelled free-form text used to
   train the mood classifier. 6 emotion classes.
2. Spotify tracks dataset (maharshipandya/spotify-tracks-dataset) -- tracks
   with numeric audio features used for K-Means clustering.

Both are fetched from the Hugging Face Hub. Run this once before training:

    python scripts/prepare_data.py

The raw files land in ``data/raw/``. They are intentionally git-ignored so the
repository stays lean; the processed/model artifacts are what the app needs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import config  # noqa: E402

EMOTION_SPLITS = {
    "train": config.EMOTION_TRAIN_CSV,
    "validation": config.EMOTION_VAL_CSV,
    "test": config.EMOTION_TEST_CSV,
}
EMOTION_BASE = (
    "https://huggingface.co/datasets/dair-ai/emotion/resolve/main/split/"
    "{split}-00000-of-00001.parquet"
)
MUSIC_URL = (
    "https://huggingface.co/datasets/maharshipandya/"
    "spotify-tracks-dataset/resolve/main/dataset.csv"
)


def download_emotion() -> None:
    """Fetch the emotion dataset splits and store them as CSV."""
    for split, out_path in EMOTION_SPLITS.items():
        print(f"[emotion] downloading {split} split ...")
        df = pd.read_parquet(EMOTION_BASE.format(split=split))
        df["emotion"] = df["label"].map(config.EMOTION_LABELS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df[["text", "label", "emotion"]].to_csv(out_path, index=False)
        print(f"[emotion] saved {len(df):,} rows -> {out_path.name}")


def download_music() -> None:
    """Fetch the Spotify tracks dataset and store it as CSV."""
    print("[music] downloading spotify tracks dataset (this may take a moment) ...")
    df = pd.read_csv(MUSIC_URL)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    config.MUSIC_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.MUSIC_CSV, index=False)
    print(f"[music] saved {len(df):,} rows -> {config.MUSIC_CSV.name}")


def main() -> None:
    download_emotion()
    download_music()
    print("\nRaw data ready in", config.RAW_DATA_DIR)


if __name__ == "__main__":
    main()
