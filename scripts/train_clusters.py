"""Train the music clusterer (StandardScaler + K-Means) and persist artifacts.

Usage:
    python scripts/train_clusters.py [--k N] [--kmin 2 --kmax 10]

Steps:
    1. Load and clean the Spotify tracks dataset.
    2. Scan candidate K values (elbow inertia + silhouette) unless --k is given.
    3. Fit K-Means on standardised audio features with the chosen K.
    4. Assign every track to a cluster and persist the scaler, model, cluster
       metadata and the clustered track table.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.music_clusterer import MusicClusterer  # noqa: E402
from src.utils import config  # noqa: E402


def load_clean_tracks() -> pd.DataFrame:
    """Load the raw Spotify dataset and apply basic data-quality cleaning."""
    df = pd.read_csv(config.MUSIC_CSV)
    before = len(df)
    df = df.dropna(subset=config.CLUSTER_FEATURES)
    # Drop obviously invalid records (zero tempo / zero duration are corrupt).
    if "tempo" in df:
        df = df[df["tempo"] > 0]
    if "duration_ms" in df:
        df = df[df["duration_ms"] > 0]
    # Deduplicate by track id, keeping the most popular copy.
    if "track_id" in df:
        sort_col = "popularity" if "popularity" in df else None
        if sort_col:
            df = df.sort_values(sort_col, ascending=False)
        df = df.drop_duplicates(subset="track_id", keep="first")
    df = df.reset_index(drop=True)
    print(f"Loaded {before:,} rows -> {len(df):,} after cleaning/dedup")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None, help="force number of clusters")
    parser.add_argument("--kmin", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=10)
    args = parser.parse_args()

    df = load_clean_tracks()

    clusterer = MusicClusterer()

    # Always scan the candidate range so the elbow/silhouette evidence is
    # recorded (and plottable), even when the final K is chosen manually.
    print("\nScanning candidate K (inertia = elbow, silhouette = separation):")
    results = clusterer.evaluate_k(df, range(args.kmin, args.kmax + 1))
    print(f"{'K':>3} {'inertia':>14} {'silhouette':>12}")
    k_scan = []
    for r in results:
        print(f"{r.k:>3} {r.inertia:>14.1f} {r.silhouette:>12.4f}")
        k_scan.append({"k": r.k, "inertia": r.inertia, "silhouette": r.silhouette})
    best_sil = max(results, key=lambda r: r.silhouette)

    if args.k is None:
        chosen_k = best_sil.k
        print(f"\nAuto-selected K={chosen_k} (highest silhouette = {best_sil.silhouette:.4f})")
    else:
        chosen_k = args.k
        print(
            f"\nUsing K={chosen_k} (silhouette peaks at K={best_sil.k}, but that "
            "trivially splits loud vs. quiet; K is set past the elbow to give the "
            "recommender enough granularity to separate mood profiles)."
        )

    clusterer.fit(df, n_clusters=chosen_k)
    clustered = clusterer.assign_clusters(df)

    # Keep only the columns needed downstream.
    keep = config.TRACK_METADATA_COLS + config.CLUSTER_FEATURES + ["cluster"]
    keep = [c for c in keep if c in clustered.columns]
    clustered = clustered[keep]

    config.MUSIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    clustered.to_parquet(config.MUSIC_TRACKS_PATH, index=False)

    centroids = clusterer.centroids_original_scale()
    sizes = clustered["cluster"].value_counts().sort_index()
    print("\n=== Cluster centroids (original units) ===")
    print(centroids.round(3).to_string())
    print("\nCluster sizes:")
    print(sizes.to_string())

    metadata = {
        "trained_on": str(date.today()),
        "dataset": "maharshipandya/spotify-tracks-dataset",
        "n_tracks": int(len(clustered)),
        "features": clusterer.features,
        "n_clusters": int(chosen_k),
        "k_scan": k_scan,
        "cluster_sizes": {int(k): int(v) for k, v in sizes.items()},
        "centroids": centroids.round(4).to_dict(orient="index"),
    }
    clusterer.save(metadata=metadata)
    print(f"\nSaved music model -> {config.MUSIC_MODEL_DIR}")


if __name__ == "__main__":
    main()
