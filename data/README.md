# Data

This project uses two public datasets. The raw files are **not committed**
(they are downloaded on demand and git-ignored). Run:

```bash
python scripts/prepare_data.py
```

to populate `data/raw/`.

## 1. Emotion text dataset — mood classification

- **Source:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) (Hugging Face)
- **Splits:** 16,000 train / 2,000 validation / 2,000 test
- **Fields:** `text` (free-form sentence), `label` (0–5), `emotion` (mapped name)
- **Classes:** sadness, joy, love, anger, fear, surprise

## 2. Spotify tracks dataset — music clustering

- **Source:** [`maharshipandya/spotify-tracks-dataset`](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) (Hugging Face)
- **Size:** 114,000 rows → 89,583 unique tracks after cleaning/dedup
- **Audio features used:** `danceability`, `energy`, `loudness`,
  `acousticness`, `valence`, `tempo`
- **Metadata kept:** `track_id`, `track_name`, `artists`, `album_name`,
  `track_genre`, `popularity`

## Directory layout

```
data/
├── raw/         # downloaded datasets (git-ignored)
└── processed/   # reserved for intermediate artifacts
```

The clustered track table the app serves from is persisted under
`models/music_clusterer/clustered_tracks.parquet`.
