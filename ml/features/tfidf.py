"""
TF-IDF Feature Extractor
==========================
Implements TF-IDF (Term Frequency-Inverse Document Frequency) feature extraction
with configurable n-grams, max features, and sublinear TF scaling.

Why TF-IDF for Spam Detection:
- Captures word importance relative to the corpus
- Spam-specific words ("free", "prize") get high weights
- Computationally efficient for traditional ML models
- Proven baseline that's hard to beat for short texts (SMS)

Performance Trade-offs:
- Unigrams: Fast, good baseline
- Bigrams: Captures phrases ("click here", "act now")
- Trigrams: Captures longer patterns but increases dimensionality
- Sublinear TF: log(1+tf) prevents long documents from dominating
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.config import MODELS_DIR

logger = logging.getLogger(__name__)


class TfidfFeatureExtractor:
    """
    Production-grade TF-IDF feature extractor.

    Wraps sklearn's TfidfVectorizer with:
    - Configurable n-gram range
    - Sublinear TF scaling (recommended for text classification)
    - Save/load for consistent training-inference behavior
    - Feature name extraction for explainability

    Usage:
        extractor = TfidfFeatureExtractor(max_features=50000, ngram_range=(1, 2))
        X_train = extractor.fit_transform(train_texts)
        X_test = extractor.transform(test_texts)
        extractor.save("tfidf_model")
    """

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
        use_idf: bool = True,
        analyzer: str = "word",
        strip_accents: str = "unicode",
        dtype: type = np.float32,
    ) -> None:
        """
        Args:
            max_features: Maximum vocabulary size. Higher = more features but slower.
            ngram_range: (min_n, max_n) for n-gram extraction.
            min_df: Minimum document frequency. Filters rare words.
            max_df: Maximum document frequency. Filters too-common words.
            sublinear_tf: Apply log(1+tf). Recommended for classification.
            use_idf: Whether to apply IDF weighting.
            analyzer: 'word' or 'char' level analysis.
            strip_accents: Unicode accent stripping.
            dtype: Output array data type (float32 for memory efficiency).
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            use_idf=use_idf,
            analyzer=analyzer,
            strip_accents=strip_accents,
            dtype=dtype,
        )
        self._is_fitted = False

        logger.info(
            "TfidfFeatureExtractor initialized: max_features=%d, "
            "ngram_range=%s, sublinear_tf=%s",
            max_features, ngram_range, sublinear_tf,
        )

    def fit(self, texts: list[str]) -> TfidfFeatureExtractor:
        """Fit the vectorizer on training texts."""
        logger.info("Fitting TF-IDF on %d texts...", len(texts))
        self.vectorizer.fit(texts)
        self._is_fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info("TF-IDF fitted. Vocabulary size: %d", vocab_size)
        return self

    def transform(self, texts: list[str]) -> csr_matrix:
        """Transform texts to TF-IDF feature matrix."""
        if not self._is_fitted:
            raise RuntimeError("TfidfFeatureExtractor is not fitted. Call fit() first.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]) -> csr_matrix:
        """Fit and transform in one step."""
        logger.info("Fit-transforming TF-IDF on %d texts...", len(texts))
        result = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        logger.info(
            "TF-IDF complete. Shape: %s, Non-zero: %d",
            result.shape, result.nnz,
        )
        return result

    def get_feature_names(self) -> list[str]:
        """Get the feature (word/n-gram) names."""
        if not self._is_fitted:
            raise RuntimeError("Extractor not fitted.")
        return self.vectorizer.get_feature_names_out().tolist()

    def get_top_features(self, n: int = 20) -> list[tuple[str, float]]:
        """
        Get top-n features by IDF score (most discriminative words).
        High IDF = rare across documents = potentially more informative.
        """
        if not self._is_fitted:
            raise RuntimeError("Extractor not fitted.")
        feature_names = self.get_feature_names()
        idf_scores = self.vectorizer.idf_
        top_indices = np.argsort(idf_scores)[-n:][::-1]
        return [(feature_names[i], float(idf_scores[i])) for i in top_indices]

    def save(self, name: str, directory: Optional[Path] = None) -> Path:
        """Save the fitted vectorizer to disk."""
        save_dir = directory or MODELS_DIR / "features"
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        logger.info("TF-IDF model saved: %s", path)
        return path

    def load(self, name: str, directory: Optional[Path] = None) -> TfidfFeatureExtractor:
        """Load a fitted vectorizer from disk."""
        load_dir = directory or MODELS_DIR / "features"
        path = load_dir / f"{name}.pkl"
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self._is_fitted = True
        logger.info("TF-IDF model loaded: %s", path)
        return self

    @property
    def vocabulary_size(self) -> int:
        """Get the size of the fitted vocabulary."""
        if not self._is_fitted:
            return 0
        return len(self.vectorizer.vocabulary_)
