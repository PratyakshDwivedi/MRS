"""Train and evaluate the mood classifier, then persist the artifacts.

Usage:
    python scripts/train_mood.py

Reads the emotion train/validation/test splits from ``data/raw`` (created by
``scripts/prepare_data.py``), fits TF-IDF + Logistic Regression on the training
split only, reports metrics on the held-out test split, and saves the
vectoriser, classifier and a metrics metadata file to ``models/mood_classifier``.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.mood_classifier import MoodClassifier  # noqa: E402
from src.utils import config  # noqa: E402


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "emotion"]).drop_duplicates(subset=["text"])
    return df


def main() -> None:
    train = _load(config.EMOTION_TRAIN_CSV)
    test = _load(config.EMOTION_TEST_CSV)
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows")
    print("Class distribution (train):")
    print(train["emotion"].value_counts().to_string())

    clf = MoodClassifier().fit(train["text"].tolist(), train["emotion"].tolist())

    y_true = test["emotion"].tolist()
    y_pred = clf.predict_batch(test["text"].tolist())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n=== Test metrics ===")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Macro Recall   : {macro_r:.4f}")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metadata = {
        "trained_on": str(date.today()),
        "dataset": "dair-ai/emotion",
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "classes": labels,
        "vectorizer": {
            "type": "TfidfVectorizer",
            "ngram_range": [1, 2],
            "max_features": 20000,
            "stop_words": "english",
            "sublinear_tf": True,
        },
        "classifier": {
            "type": "LogisticRegression",
            "C": 5.0,
            "class_weight": "balanced",
        },
        "metrics": {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
    }

    clf.save(metadata=metadata)
    print(f"\nSaved mood model -> {config.MOOD_MODEL_DIR}")
    print(json.dumps(metadata["metrics"], indent=2))


if __name__ == "__main__":
    main()
