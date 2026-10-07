"""Supervised mood classification: TF-IDF + Logistic Regression.

Pipeline design notes
---------------------
* The text is cleaned (``text_preprocessor.clean_text``) and vectorised with a
  TF-IDF vectoriser using unigrams + bigrams. TF-IDF up-weights words that are
  distinctive to a document while down-weighting terms common across the whole
  corpus, which suits short emotional sentences.
* Logistic Regression is the classifier: a strong, interpretable linear
  baseline that handles high-dimensional sparse TF-IDF features efficiently.
  ``class_weight="balanced"`` compensates for the class imbalance in the
  emotion dataset (joy/sadness dominate, surprise is rare).
* The vectoriser is fitted ONLY on training data to avoid leakage. Both the
  vectoriser and the classifier are persisted so inference never retrains.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.preprocessing.text_preprocessor import clean_text
from src.utils import config


@dataclass
class MoodPrediction:
    """A single mood prediction with class probabilities."""

    mood: str
    confidence: float
    probabilities: dict[str, float]


class MoodClassifier:
    """TF-IDF + Logistic Regression mood classifier."""

    def __init__(
        self,
        vectorizer: TfidfVectorizer | None = None,
        classifier: LogisticRegression | None = None,
    ) -> None:
        self.vectorizer = vectorizer
        self.classifier = classifier

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self,
        texts: list[str],
        labels: list[str],
        *,
        max_features: int = 20_000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        C: float = 5.0,
    ) -> "MoodClassifier":
        """Fit the vectoriser and classifier on training data only.

        Args:
            texts: Raw training texts.
            labels: Emotion label per text.
            max_features: TF-IDF vocabulary cap.
            ngram_range: TF-IDF n-gram range (unigrams + bigrams by default).
            min_df: Ignore terms appearing in fewer than this many documents.
            C: Inverse L2 regularisation strength for Logistic Regression.
        """
        cleaned = [clean_text(t) for t in texts]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=True,
        )
        features = self.vectorizer.fit_transform(cleaned)

        self.classifier = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        )
        self.classifier.fit(features, labels)
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> None:
        if self.vectorizer is None or self.classifier is None:
            raise RuntimeError(
                "MoodClassifier is not trained/loaded. Train it or call "
                "MoodClassifier.load() first."
            )

    def predict(self, text: str) -> MoodPrediction:
        """Predict the mood of a single text string."""
        self._ensure_loaded()
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("Input text is empty after cleaning.")
        features = self.vectorizer.transform([cleaned])
        proba = self.classifier.predict_proba(features)[0]
        classes = list(self.classifier.classes_)
        probabilities = {c: float(p) for c, p in zip(classes, proba)}
        best_idx = int(proba.argmax())
        return MoodPrediction(
            mood=classes[best_idx],
            confidence=float(proba[best_idx]),
            probabilities=probabilities,
        )

    def predict_batch(self, texts: list[str]) -> list[str]:
        """Predict moods for many texts (used during evaluation)."""
        self._ensure_loaded()
        cleaned = [clean_text(t) for t in texts]
        features = self.vectorizer.transform(cleaned)
        return list(self.classifier.predict(features))

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, metadata: dict | None = None) -> None:
        """Persist the vectoriser, classifier and optional metadata."""
        self._ensure_loaded()
        config.MOOD_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, config.MOOD_VECTORIZER_PATH)
        joblib.dump(self.classifier, config.MOOD_CLASSIFIER_PATH)
        if metadata is not None:
            config.MOOD_METADATA_PATH.write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

    @classmethod
    def load(
        cls,
        vectorizer_path: Path | None = None,
        classifier_path: Path | None = None,
    ) -> "MoodClassifier":
        """Load a persisted classifier from disk."""
        vectorizer_path = vectorizer_path or config.MOOD_VECTORIZER_PATH
        classifier_path = classifier_path or config.MOOD_CLASSIFIER_PATH
        if not vectorizer_path.exists() or not classifier_path.exists():
            raise FileNotFoundError(
                "Mood model artifacts not found. Train first with "
                "`python scripts/train_mood.py`."
            )
        return cls(
            vectorizer=joblib.load(vectorizer_path),
            classifier=joblib.load(classifier_path),
        )
