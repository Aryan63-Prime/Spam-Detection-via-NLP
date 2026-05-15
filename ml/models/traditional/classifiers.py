"""
Traditional ML Models for Spam Classification
===============================================
Implements all 6 traditional ML classifiers with the BaseModel interface.

Each model includes:
- Configurable hyperparameters
- Class weight balancing for imbalanced data
- Probability calibration
- Save/load support

Model Comparison (SMS Spam Collection):
┌──────────────────────┬──────────┬──────┬──────────┐
│ Model                │ Speed    │ F1   │ Use Case │
├──────────────────────┼──────────┼──────┼──────────┤
│ Naive Bayes          │ Fastest  │ ~95% │ Baseline │
│ Logistic Regression  │ Fast     │ ~97% │ Prod     │
│ SVM                  │ Medium   │ ~97% │ Prod     │
│ Random Forest        │ Slow     │ ~96% │ Ensemble │
│ XGBoost              │ Medium   │ ~98% │ Best     │
│ LightGBM             │ Fast     │ ~98% │ Best     │
└──────────────────────┴──────────┴──────┴──────────┘
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from ml.config import TRADITIONAL_ML_DEFAULTS
from ml.models.base_model import BaseModel

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Naive Bayes
# ──────────────────────────────────────────────

class NaiveBayesModel(BaseModel):
    """
    Multinomial Naive Bayes classifier.

    Theory:
    - Based on Bayes' theorem with the naive independence assumption
    - P(spam|features) ∝ P(features|spam) × P(spam)
    - Each feature (word) is treated independently

    Why for spam:
    - Extremely fast training and inference
    - Works well with TF-IDF features
    - Strong baseline for text classification
    - Handles high-dimensional sparse data naturally

    Limitations:
    - Independence assumption rarely holds
    - Can't capture word interactions
    - Requires non-negative features (TF-IDF is fine)
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(name="naive_bayes", model_type="traditional")
        params = {**TRADITIONAL_ML_DEFAULTS.naive_bayes, **kwargs}
        self.model = MultinomialNB(**params)
        self.metadata.hyperparameters = params

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> dict:
        start = time.perf_counter()
        self.model.fit(X_train, y_train)
        self._is_trained = True
        train_time = time.perf_counter() - start

        self.metadata.training_time_seconds = train_time
        self.metadata.training_samples = X_train.shape[0]

        metrics = {"training_time": train_time}
        logger.info("NaiveBayes trained in %.3fs on %d samples.", train_time, X_train.shape[0])
        return metrics

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def _get_model_object(self) -> Any:
        return self.model

    def _set_model_object(self, model: Any) -> None:
        self.model = model


# ──────────────────────────────────────────────
# Logistic Regression
# ──────────────────────────────────────────────

class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression classifier.

    Theory:
    - Models P(spam|x) using sigmoid function: σ(wᵀx + b)
    - Learns linear decision boundary in feature space
    - L2 regularization (default) prevents overfitting

    Why for spam:
    - Fast, interpretable, and well-calibrated probabilities
    - Feature weights directly show word importance
    - Works excellently with TF-IDF
    - class_weight="balanced" handles imbalance

    Production recommendation:
    - Best balance of speed, accuracy, and interpretability
    - Often the go-to production model for text classification
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(name="logistic_regression", model_type="traditional")
        params = {**TRADITIONAL_ML_DEFAULTS.logistic_regression, **kwargs}
        self.model = LogisticRegression(**params)
        self.metadata.hyperparameters = params

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> dict:
        start = time.perf_counter()
        self.model.fit(X_train, y_train)
        self._is_trained = True
        train_time = time.perf_counter() - start

        self.metadata.training_time_seconds = train_time
        self.metadata.training_samples = X_train.shape[0]

        logger.info("LogisticRegression trained in %.3fs.", train_time)
        return {"training_time": train_time}

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: list[str], top_n: int = 20) -> list[tuple[str, float]]:
        """Get top features by coefficient magnitude (interpretability)."""
        coefficients = self.model.coef_[0]
        top_indices = np.argsort(np.abs(coefficients))[-top_n:][::-1]
        return [(feature_names[i], float(coefficients[i])) for i in top_indices]

    def _get_model_object(self) -> Any:
        return self.model

    def _set_model_object(self, model: Any) -> None:
        self.model = model


# ──────────────────────────────────────────────
# SVM
# ──────────────────────────────────────────────

class SVMModel(BaseModel):
    """
    Support Vector Machine classifier.

    Theory:
    - Finds the maximum-margin hyperplane separating classes
    - Kernel trick maps to higher dimensions (we use linear for text)
    - Support vectors define the decision boundary

    Why LinearSVC:
    - Linear kernel is optimal for high-dimensional sparse text data
    - Much faster than RBF kernel for TF-IDF features
    - Scales to large vocabularies
    - CalibratedClassifierCV adds probability support
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(name="svm", model_type="traditional")
        svm_params = {**TRADITIONAL_ML_DEFAULTS.svm, **kwargs}
        # LinearSVC doesn't support probability, so we wrap it
        svm_params.pop("probability", None)
        svm_params.pop("random_state", None)
        base_svm = LinearSVC(
            C=svm_params.get("C", 1.0),
            class_weight=svm_params.get("class_weight", "balanced"),
            max_iter=5000,
            dual="auto",
        )
        self.model = CalibratedClassifierCV(base_svm, cv=3)
        self.metadata.hyperparameters = svm_params

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> dict:
        start = time.perf_counter()
        self.model.fit(X_train, y_train)
        self._is_trained = True
        train_time = time.perf_counter() - start

        self.metadata.training_time_seconds = train_time
        self.metadata.training_samples = X_train.shape[0]

        logger.info("SVM trained in %.3fs.", train_time)
        return {"training_time": train_time}

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def _get_model_object(self) -> Any:
        return self.model

    def _set_model_object(self, model: Any) -> None:
        self.model = model


# ──────────────────────────────────────────────
# Random Forest
# ──────────────────────────────────────────────

class RandomForestModel(BaseModel):
    """
    Random Forest classifier.

    Theory:
    - Ensemble of decision trees trained on bootstrap samples
    - Each tree uses a random subset of features
    - Final prediction via majority vote
    - Reduces overfitting through averaging

    Why for spam:
    - Handles non-linear patterns
    - Feature importance built-in
    - Robust to noisy features
    - No feature scaling needed

    Limitations:
    - Slower inference than linear models
    - Higher memory usage
    - Less interpretable than LR
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(name="random_forest", model_type="traditional")
        params = {**TRADITIONAL_ML_DEFAULTS.random_forest, **kwargs}
        self.model = RandomForestClassifier(**params)
        self.metadata.hyperparameters = params

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> dict:
        start = time.perf_counter()
        self.model.fit(X_train, y_train)
        self._is_trained = True
        train_time = time.perf_counter() - start

        self.metadata.training_time_seconds = train_time
        self.metadata.training_samples = X_train.shape[0]

        logger.info("RandomForest trained in %.3fs.", train_time)
        return {"training_time": train_time}

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: list[str], top_n: int = 20) -> list[tuple[str, float]]:
        """Get top features by Gini importance."""
        importances = self.model.feature_importances_
        top_indices = np.argsort(importances)[-top_n:][::-1]
        return [(feature_names[i], float(importances[i])) for i in top_indices]

    def _get_model_object(self) -> Any:
        return self.model

    def _set_model_object(self, model: Any) -> None:
        self.model = model


# ──────────────────────────────────────────────
# XGBoost
# ──────────────────────────────────────────────

class XGBoostModel(BaseModel):
    """
    XGBoost (Extreme Gradient Boosting) classifier.

    Theory:
    - Sequential ensemble: each tree corrects errors of the previous
    - Uses gradient descent to minimize loss function
    - Regularization (L1/L2) prevents overfitting
    - Handles missing values natively

    Why for spam:
    - State-of-the-art for tabular/sparse data
    - Excellent with TF-IDF features
    - Built-in feature importance
    - GPU acceleration available
    - Robust to overfitting with proper regularization
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(name="xgboost", model_type="traditional")
        import xgboost as xgb
        params = {**TRADITIONAL_ML_DEFAULTS.xgboost, **kwargs}
        self.model = xgb.XGBClassifier(**params)
        self.metadata.hyperparameters = params

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> dict:
        start = time.perf_counter()

        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        self.model.fit(X_train, y_train, **fit_params)
        self._is_trained = True
        train_time = time.perf_counter() - start

        self.metadata.training_time_seconds = train_time
        self.metadata.training_samples = X_train.shape[0]

        logger.info("XGBoost trained in %.3fs.", train_time)
        return {"training_time": train_time}

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: list[str], top_n: int = 20) -> list[tuple[str, float]]:
        """Get top features by gain importance."""
        importances = self.model.feature_importances_
        top_indices = np.argsort(importances)[-top_n:][::-1]
        return [(feature_names[i], float(importances[i])) for i in top_indices]

    def _get_model_object(self) -> Any:
        return self.model

    def _set_model_object(self, model: Any) -> None:
        self.model = model


# ──────────────────────────────────────────────
# LightGBM
# ──────────────────────────────────────────────

class LightGBMModel(BaseModel):
    """
    LightGBM classifier.

    Theory:
    - Gradient boosting with leaf-wise tree growth (vs. level-wise)
    - GOSS: Gradient-based One-Side Sampling for speed
    - EFB: Exclusive Feature Bundling for sparse features
    - Histogram-based split finding

    Why for spam:
    - Faster than XGBoost on large datasets
    - Excellent with sparse TF-IDF features (EFB)
    - Lower memory usage
    - Native categorical feature support

    Production note:
    - Often matches XGBoost accuracy with 2-5x faster training
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(name="lightgbm", model_type="traditional")
        import lightgbm as lgb
        params = {**TRADITIONAL_ML_DEFAULTS.lightgbm, **kwargs}
        self.model = lgb.LGBMClassifier(**params)
        self.metadata.hyperparameters = params

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> dict:
        start = time.perf_counter()

        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]

        self.model.fit(X_train, y_train, **fit_params)
        self._is_trained = True
        train_time = time.perf_counter() - start

        self.metadata.training_time_seconds = train_time
        self.metadata.training_samples = X_train.shape[0]

        logger.info("LightGBM trained in %.3fs.", train_time)
        return {"training_time": train_time}

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: list[str], top_n: int = 20) -> list[tuple[str, float]]:
        """Get top features by gain importance."""
        importances = self.model.feature_importances_
        top_indices = np.argsort(importances)[-top_n:][::-1]
        return [(feature_names[i], float(importances[i])) for i in top_indices]

    def _get_model_object(self) -> Any:
        return self.model

    def _set_model_object(self, model: Any) -> None:
        self.model = model


# ──────────────────────────────────────────────
# Model Registry (Factory)
# ──────────────────────────────────────────────

TRADITIONAL_MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "naive_bayes": NaiveBayesModel,
    "logistic_regression": LogisticRegressionModel,
    "svm": SVMModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
}


def create_traditional_model(name: str, **kwargs) -> BaseModel:
    """
    Factory function to create a traditional ML model by name.

    Args:
        name: Model name (e.g., "xgboost", "naive_bayes").

    Returns:
        An instance of the requested model.
    """
    if name not in TRADITIONAL_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{name}'. "
            f"Available: {list(TRADITIONAL_MODEL_REGISTRY.keys())}"
        )
    return TRADITIONAL_MODEL_REGISTRY[name](**kwargs)
