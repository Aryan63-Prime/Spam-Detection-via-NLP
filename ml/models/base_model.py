"""
Base Model Interface
=====================
Abstract base class enforcing a consistent API across all models.

Architecture Decision:
- All models (Traditional, DL, Transformer) implement this interface
- Guarantees train/predict/evaluate/save/load work uniformly
- Enables model-agnostic evaluation and benchmarking
- Follows the Open/Closed principle (open for extension, closed for modification)
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from ml.config import MODELS_DIR

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata associated with a trained model."""
    model_name: str = ""
    model_type: str = ""  # "traditional", "deep_learning", "transformer"
    version: str = "1.0.0"
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    training_time_seconds: float = 0.0
    training_samples: int = 0
    hyperparameters: dict = field(default_factory=dict)
    feature_type: str = ""  # "tfidf", "embeddings", etc.


class BaseModel(ABC):
    """
    Abstract base class for all spam classification models.

    All models in this project MUST implement this interface.
    This ensures consistent behavior across:
    - Traditional ML (NB, LR, SVM, RF, XGB, LGBM)
    - Deep Learning (CNN, LSTM, GRU)
    - Transformers (BERT, DistilBERT, RoBERTa)
    """

    def __init__(self, name: str, model_type: str = "traditional") -> None:
        self.name = name
        self.model_type = model_type
        self.metadata = ModelMetadata(model_name=name, model_type=model_type)
        self._is_trained = False

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> dict:
        """
        Train the model.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Optional validation features.
            y_val: Optional validation labels.

        Returns:
            Dictionary with training metrics.
        """
        ...

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Input features.

        Returns:
            Array of predicted labels (0=ham, 1=spam).
        """
        ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """
        Generate prediction probabilities.

        Args:
            X: Input features.

        Returns:
            Array of shape (n_samples, 2) with [P(ham), P(spam)].
        """
        ...

    def save(self, directory: Optional[Path] = None) -> Path:
        """
        Save the trained model and metadata to disk.

        Default implementation uses pickle. Override for
        custom serialization (e.g., PyTorch state_dict).
        """
        save_dir = directory or MODELS_DIR / self.name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = save_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self._get_model_object(), f)

        # Save metadata
        meta_path = save_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(asdict(self.metadata), f, indent=2, default=str)

        logger.info("Model saved: %s → %s", self.name, save_dir)
        return save_dir

    def load(self, directory: Optional[Path] = None) -> BaseModel:
        """Load a trained model from disk."""
        load_dir = directory or MODELS_DIR / self.name

        model_path = load_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, "rb") as f:
            model_obj = pickle.load(f)
        self._set_model_object(model_obj)
        self._is_trained = True

        # Load metadata
        meta_path = load_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta_dict = json.load(f)
            self.metadata = ModelMetadata(**meta_dict)

        logger.info("Model loaded: %s ← %s", self.name, load_dir)
        return self

    @abstractmethod
    def _get_model_object(self) -> Any:
        """Return the internal model object for serialization."""
        ...

    @abstractmethod
    def _set_model_object(self, model: Any) -> None:
        """Set the internal model object from deserialization."""
        ...

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return f"{self.__class__.__name__}(name='{self.name}', status={status})"
