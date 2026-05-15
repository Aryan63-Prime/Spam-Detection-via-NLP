"""
Model Evaluation & Metrics Module
====================================
Comprehensive evaluation suite for spam classifiers.

Implements:
- Standard metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix analysis
- Per-class metrics
- Benchmark comparison across models
- Error analysis

Why these metrics for spam detection:
- Accuracy alone is misleading (90% accuracy possible by predicting all ham)
- Precision: Of predicted spam, how many are actually spam? (false positive cost)
- Recall: Of actual spam, how many did we catch? (missed spam cost)
- F1: Harmonic mean balances precision and recall
- ROC-AUC: Threshold-independent performance measure
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.config import LABEL_MAP_INV

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for all evaluation metrics of a single model."""
    model_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    confusion_matrix: list = field(default_factory=list)
    classification_report: str = ""
    per_class_metrics: dict = field(default_factory=dict)
    training_time: float = 0.0
    inference_time_per_sample_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self) -> str:
        """One-line summary for benchmark tables."""
        return (
            f"{self.model_name:<25} | "
            f"Acc: {self.accuracy:.4f} | "
            f"P: {self.precision:.4f} | "
            f"R: {self.recall:.4f} | "
            f"F1: {self.f1:.4f} | "
            f"AUC: {self.roc_auc:.4f}"
        )


class ModelEvaluator:
    """
    Production-grade model evaluation suite.

    Usage:
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(model, X_test, y_test)
        print(result.summary_line())

        # Benchmark multiple models
        results = evaluator.benchmark(models, X_test, y_test)
        evaluator.print_benchmark_table(results)
    """

    def __init__(self, average: str = "binary", pos_label: int = 1) -> None:
        """
        Args:
            average: Averaging strategy for multi-class metrics.
            pos_label: The positive class label (1=spam).
        """
        self.average = average
        self.pos_label = pos_label

    def evaluate(
        self,
        model,
        X_test,
        y_test,
        model_name: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a trained model on test data.

        Args:
            model: Trained model with predict() and predict_proba().
            X_test: Test features.
            y_test: True labels.
            model_name: Name for the result.

        Returns:
            EvaluationResult with all metrics.
        """
        import time

        name = model_name or getattr(model, "name", "unknown")

        # Predictions
        start = time.perf_counter()
        y_pred = model.predict(X_test)
        inference_time = time.perf_counter() - start

        # Probabilities (for ROC-AUC)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = y_pred.astype(float)

        y_test_np = np.array(y_test)

        # Core metrics
        acc = accuracy_score(y_test_np, y_pred)
        prec = precision_score(y_test_np, y_pred, pos_label=self.pos_label, zero_division=0)
        rec = recall_score(y_test_np, y_pred, pos_label=self.pos_label, zero_division=0)
        f1 = f1_score(y_test_np, y_pred, pos_label=self.pos_label, zero_division=0)

        try:
            roc = roc_auc_score(y_test_np, y_proba)
        except ValueError:
            roc = 0.0

        # Confusion matrix
        cm = confusion_matrix(y_test_np, y_pred).tolist()

        # Classification report
        target_names = [LABEL_MAP_INV.get(i, str(i)) for i in sorted(set(y_test_np))]
        report = classification_report(
            y_test_np, y_pred, target_names=target_names, zero_division=0
        )

        # Per-class metrics
        per_class = {}
        for label_id, label_name in LABEL_MAP_INV.items():
            mask = y_test_np == label_id
            if mask.sum() > 0:
                per_class[label_name] = {
                    "support": int(mask.sum()),
                    "precision": float(precision_score(
                        y_test_np == label_id, y_pred == label_id, zero_division=0
                    )),
                    "recall": float(recall_score(
                        y_test_np == label_id, y_pred == label_id, zero_division=0
                    )),
                }

        n_samples = X_test.shape[0] if hasattr(X_test, "shape") else len(X_test)
        inference_per_sample = (inference_time / max(n_samples, 1)) * 1000

        result = EvaluationResult(
            model_name=name,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=roc,
            confusion_matrix=cm,
            classification_report=report,
            per_class_metrics=per_class,
            inference_time_per_sample_ms=inference_per_sample,
        )

        logger.info("Evaluation complete: %s", result.summary_line())
        return result

    def benchmark(
        self,
        models: list,
        X_test,
        y_test,
    ) -> list[EvaluationResult]:
        """
        Benchmark multiple models on the same test set.

        Args:
            models: List of trained models.
            X_test: Test features.
            y_test: True labels.

        Returns:
            List of EvaluationResult, sorted by F1 score (descending).
        """
        results = []
        for model in models:
            try:
                result = self.evaluate(model, X_test, y_test)
                results.append(result)
            except Exception as e:
                logger.error("Failed to evaluate %s: %s", model.name, e)

        # Sort by F1 score
        results.sort(key=lambda r: r.f1, reverse=True)
        return results

    @staticmethod
    def print_benchmark_table(results: list[EvaluationResult]) -> None:
        """Print a formatted benchmark comparison table."""
        header = f"{'Model':<25} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'AUC':>7} | {'ms/sample':>9}"
        separator = "-" * len(header)

        print(f"\n{separator}")
        print("MODEL BENCHMARK RESULTS")
        print(separator)
        print(header)
        print(separator)

        for r in results:
            print(
                f"{r.model_name:<25} | "
                f"{r.accuracy:>7.4f} | "
                f"{r.precision:>7.4f} | "
                f"{r.recall:>7.4f} | "
                f"{r.f1:>7.4f} | "
                f"{r.roc_auc:>7.4f} | "
                f"{r.inference_time_per_sample_ms:>8.3f}ms"
            )

        print(separator)
        print(f"Best model by F1: {results[0].model_name} ({results[0].f1:.4f})")
        print()

    @staticmethod
    def save_results(
        results: list[EvaluationResult],
        path: Path,
    ) -> None:
        """Save benchmark results to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Benchmark results saved: %s", path)
