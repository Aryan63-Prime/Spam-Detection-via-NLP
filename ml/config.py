"""
ML Configuration Module
========================
Centralized configuration constants for the ML pipeline.
Separated from the application config (configs/settings.py)
to keep ML concerns isolated.

This module defines:
- Model hyperparameter defaults
- Supported model names/enums
- Dataset paths and URLs
- Feature engineering constants
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# ──────────────────────────────────────────────
# Path Constants
# ──────────────────────────────────────────────
ML_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ML_ROOT.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DATA_DIR = DATASETS_DIR / "raw"
PROCESSED_DATA_DIR = DATASETS_DIR / "processed"
CACHE_DIR = DATASETS_DIR / "cache"


class ModelType(str, Enum):
    """Supported model types."""
    # Traditional ML
    NAIVE_BAYES = "naive_bayes"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

    # Deep Learning
    TEXT_CNN = "text_cnn"
    LSTM = "lstm"
    BILSTM = "bilstm"
    GRU = "gru"

    # Transformers
    BERT = "bert"
    DISTILBERT = "distilbert"
    ROBERTA = "roberta"
    DEBERTA = "deberta"


class FeatureType(str, Enum):
    """Supported feature extraction methods."""
    TFIDF = "tfidf"
    BOW = "bow"
    WORD2VEC = "word2vec"
    FASTTEXT = "fasttext"
    GLOVE = "glove"
    SENTENCE_BERT = "sentence_bert"
    TRANSFORMER = "transformer"


class DatasetSource(str, Enum):
    """Known dataset sources."""
    SMS_SPAM = "sms_spam"
    ENRON = "enron"
    SPAM_ASSASSIN = "spam_assassin"
    CUSTOM = "custom"


# ──────────────────────────────────────────────
# Dataset URLs (Public Datasets)
# ──────────────────────────────────────────────
DATASET_URLS: dict[str, str] = {
    "sms_spam": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00228/smsspamcollection.zip"
    ),
}

# ──────────────────────────────────────────────
# Label Encoding
# ──────────────────────────────────────────────
LABEL_MAP: dict[str, int] = {"ham": 0, "spam": 1}
LABEL_MAP_INV: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

# ──────────────────────────────────────────────
# Model Hyperparameter Defaults
# ──────────────────────────────────────────────

@dataclass
class TraditionalMLDefaults:
    """Default hyperparameters for traditional ML models."""

    naive_bayes: dict = field(default_factory=lambda: {
        "alpha": 1.0,
        "fit_prior": True,
    })

    logistic_regression: dict = field(default_factory=lambda: {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "random_state": 42,
    })

    svm: dict = field(default_factory=lambda: {
        "C": 1.0,
        "kernel": "linear",
        "class_weight": "balanced",
        "probability": True,
        "random_state": 42,
    })

    random_forest: dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    })

    xgboost: dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    })

    lightgbm: dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": -1,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": 42,
        "verbose": -1,
    })


@dataclass
class DeepLearningDefaults:
    """Default hyperparameters for deep learning models."""
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 1e-3
    early_stopping_patience: int = 3
    max_sequence_length: int = 256
    vocab_size: int = 30000


@dataclass
class TransformerDefaults:
    """Default hyperparameters for transformer models."""
    model_names: dict = field(default_factory=lambda: {
        "bert": "bert-base-uncased",
        "distilbert": "distilbert-base-uncased",
        "roberta": "roberta-base",
        "deberta": "microsoft/deberta-v3-small",
    })
    max_length: int = 256
    batch_size: int = 16  # Smaller for GPU memory
    epochs: int = 5
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = False  # Set True when GPU available
    gradient_accumulation_steps: int = 2


# Singleton defaults
TRADITIONAL_ML_DEFAULTS = TraditionalMLDefaults()
DEEP_LEARNING_DEFAULTS = DeepLearningDefaults()
TRANSFORMER_DEFAULTS = TransformerDefaults()
