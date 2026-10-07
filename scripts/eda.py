"""Generate exploratory-data-analysis figures used in the README.

Produces PNGs in ``assets/`` from the real datasets and the trained clusterer:

    * emotion_class_distribution.png  -- class balance of the emotion dataset
    * text_length_distribution.png    -- token-count distribution of the texts
    * feature_correlation.png         -- correlation of Spotify audio features
    * k_selection.png                 -- elbow (inertia) + silhouette vs K
    * cluster_profiles.png            -- standardised centroid heatmap

Usage:
    python scripts/eda.py

Run after prepare_data.py and train_clusters.py so the model metadata exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import config  # noqa: E402

PALETTE = "#6C5CE7"


def _save(fig, name: str) -> None:
    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.ASSETS_DIR / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", out.name)


def emotion_distribution() -> None:
    df = pd.read_csv(config.EMOTION_TRAIN_CSV)
    counts = df["emotion"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color=PALETTE)
    ax.set_title("Emotion class distribution (training set)")
    ax.set_ylabel("samples")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 50, f"{v:,}", ha="center", fontsize=9)
    _save(fig, "emotion_class_distribution.png")


def text_length() -> None:
    df = pd.read_csv(config.EMOTION_TRAIN_CSV)
    lengths = df["text"].astype(str).str.split().map(len)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=40, color=PALETTE, alpha=0.85)
    ax.axvline(lengths.median(), color="#E17055", linestyle="--",
               label=f"median = {int(lengths.median())} words")
    ax.set_title("Text length distribution")
    ax.set_xlabel("words per sample")
    ax.set_ylabel("frequency")
    ax.legend()
    _save(fig, "text_length_distribution.png")


def feature_correlation() -> None:
    df = pd.read_csv(config.MUSIC_CSV)
    corr = df[config.CLUSTER_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(config.CLUSTER_FEATURES)))
    ax.set_yticks(range(len(config.CLUSTER_FEATURES)))
    ax.set_xticklabels(config.CLUSTER_FEATURES, rotation=45, ha="right")
    ax.set_yticklabels(config.CLUSTER_FEATURES)
    for i in range(len(config.CLUSTER_FEATURES)):
        for j in range(len(config.CLUSTER_FEATURES)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_title("Audio feature correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "feature_correlation.png")


def k_selection() -> None:
    meta = json.loads(config.MUSIC_METADATA_PATH.read_text(encoding="utf-8"))
    scan = meta.get("k_scan")
    if not scan:
        print("no k_scan in metadata (model trained with forced --k); skipping k_selection")
        return
    ks = [r["k"] for r in scan]
    inertia = [r["inertia"] for r in scan]
    sil = [r["silhouette"] for r in scan]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ks, inertia, "o-", color=PALETTE, label="inertia (elbow)")
    ax1.set_xlabel("K")
    ax1.set_ylabel("inertia (WCSS)", color=PALETTE)
    ax2 = ax1.twinx()
    ax2.plot(ks, sil, "s--", color="#E17055", label="silhouette")
    ax2.set_ylabel("silhouette", color="#E17055")
    ax1.set_title("K selection: elbow + silhouette")
    _save(fig, "k_selection.png")


def cluster_profiles() -> None:
    meta = json.loads(config.MUSIC_METADATA_PATH.read_text(encoding="utf-8"))
    centroids = meta["centroids"]
    features = meta["features"]
    clusters = sorted(int(k) for k in centroids)
    mat = np.array([[centroids[str(c)][f] for f in features] for c in clusters])
    # z-score each feature column so the heatmap shows relative high/low.
    z = (mat - mat.mean(axis=0)) / (mat.std(axis=0) + 1e-9)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(z, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"cluster {c}" for c in clusters])
    for i in range(len(clusters)):
        for j in range(len(features)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Cluster audio profiles (centroids, colour = relative z-score)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "cluster_profiles.png")


def main() -> None:
    emotion_distribution()
    text_length()
    feature_correlation()
    k_selection()
    cluster_profiles()
    print("\nEDA figures written to", config.ASSETS_DIR)


if __name__ == "__main__":
    main()
