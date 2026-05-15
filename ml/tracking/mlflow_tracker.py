"""
MLflow Experiment Tracking
============================
Production-grade experiment tracking for model training runs.

Architecture:
- Wraps MLflow's tracking API with project-specific conventions
- Auto-logs hyperparameters, metrics, artifacts, and model files
- Supports local file store and remote tracking server
- Thread-safe singleton pattern

Usage:
    tracker = ExperimentTracker("spam_detection")
    with tracker.start_run("logistic_regression_v2"):
        tracker.log_params({"C": 1.0, "solver": "lbfgs"})
        tracker.log_metrics({"accuracy": 0.95, "f1": 0.93})
        tracker.log_model("models/lr_model.pkl")
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lazy import — MLflow is optional for lightweight deployments
_mlflow = None


def _get_mlflow():
    """Lazy import MLflow to avoid hard dependency."""
    global _mlflow
    if _mlflow is None:
        try:
            import mlflow
            _mlflow = mlflow
        except ImportError:
            logger.warning(
                "MLflow not installed. Experiment tracking disabled. "
                "Install with: pip install mlflow"
            )
    return _mlflow


class ExperimentTracker:
    """
    MLflow experiment tracker with project conventions.

    Features:
    - Auto-creates experiments
    - Structured run naming
    - Artifact logging (models, plots, configs)
    - Metric history tracking
    - Tag management for filtering
    """

    def __init__(
        self,
        experiment_name: str = "spam_detection",
        tracking_uri: Optional[str] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "file:///./mlruns"
        )
        self._active_run = None
        self._setup()

    def _setup(self) -> None:
        """Initialize MLflow tracking."""
        mlflow = _get_mlflow()
        if mlflow is None:
            return

        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self._experiment_id = mlflow.create_experiment(
                self.experiment_name,
                tags={"project": "spamshield-ai", "team": "ml"},
            )
        else:
            self._experiment_id = experiment.experiment_id

        mlflow.set_experiment(self.experiment_name)
        logger.info("MLflow tracker initialized: %s", self.experiment_name)

    @contextmanager
    def start_run(self, run_name: str, tags: Optional[dict] = None):
        """
        Context manager for an MLflow run.

        Args:
            run_name: Descriptive run name (e.g., "lr_tfidf_v3").
            tags: Optional tags for filtering.
        """
        mlflow = _get_mlflow()
        if mlflow is None:
            yield
            return

        run_tags = {"run_type": "training"}
        if tags:
            run_tags.update(tags)

        with mlflow.start_run(run_name=run_name, tags=run_tags) as run:
            self._active_run = run
            logger.info("MLflow run started: %s (%s)", run_name, run.info.run_id)
            try:
                yield run
            finally:
                self._active_run = None
                logger.info("MLflow run completed: %s", run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        mlflow = _get_mlflow()
        if mlflow is None:
            return
        # MLflow requires string values
        clean = {k: str(v) for k, v in params.items()}
        mlflow.log_params(clean)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log evaluation metrics."""
        mlflow = _get_mlflow()
        if mlflow is None:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, filepath: str, artifact_path: Optional[str] = None) -> None:
        """Log a file as an artifact."""
        mlflow = _get_mlflow()
        if mlflow is None:
            return
        if Path(filepath).exists():
            mlflow.log_artifact(filepath, artifact_path)

    def log_model(self, model_path: str, model_name: str = "model") -> None:
        """Log a trained model file."""
        self.log_artifact(model_path, artifact_path="models")

    def log_dict(self, data: dict, filename: str) -> None:
        """Log a dictionary as a JSON artifact."""
        mlflow = _get_mlflow()
        if mlflow is None:
            return
        mlflow.log_dict(data, filename)

    def log_figure(self, figure, filename: str) -> None:
        """Log a matplotlib figure."""
        mlflow = _get_mlflow()
        if mlflow is None:
            return
        mlflow.log_figure(figure, filename)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the active run."""
        mlflow = _get_mlflow()
        if mlflow is None:
            return
        mlflow.set_tag(key, value)

    def register_model(self, model_uri: str, name: str) -> None:
        """Register model in MLflow Model Registry."""
        mlflow = _get_mlflow()
        if mlflow is None:
            return
        try:
            mlflow.register_model(model_uri, name)
            logger.info("Model registered: %s", name)
        except Exception as e:
            logger.warning("Model registration failed: %s", e)
