"""
Prometheus Metrics Module
===========================
Custom metrics for API and ML monitoring.

Metrics:
- HTTP request duration (histogram)
- HTTP request count (counter)
- Active predictions (gauge)
- Model prediction latency (histogram)
- Prediction label distribution (counter)
- Model confidence distribution (histogram)

Integration:
- Mounted at /metrics endpoint in FastAPI
- Scraped by Prometheus every 15s
- Visualized in Grafana dashboards
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Lazy import — prometheus_client is optional
_prom = None


def _get_prometheus():
    global _prom
    if _prom is None:
        try:
            import prometheus_client
            _prom = prometheus_client
        except ImportError:
            logger.info("prometheus_client not installed. Metrics disabled.")
    return _prom


class MetricsCollector:
    """
    Centralized Prometheus metrics for SpamShield AI.

    All metrics follow naming convention:
    spamshield_{subsystem}_{metric_name}_{unit}
    """

    def __init__(self) -> None:
        prom = _get_prometheus()
        if prom is None:
            self._enabled = False
            return

        self._enabled = True

        # HTTP Metrics
        self.http_requests_total = prom.Counter(
            "spamshield_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
        )
        self.http_request_duration = prom.Histogram(
            "spamshield_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )

        # ML Metrics
        self.predictions_total = prom.Counter(
            "spamshield_predictions_total",
            "Total predictions made",
            ["model", "prediction"],
        )
        self.prediction_latency = prom.Histogram(
            "spamshield_prediction_latency_ms",
            "Model prediction latency in milliseconds",
            ["model"],
            buckets=(5, 10, 25, 50, 100, 250, 500, 1000),
        )
        self.prediction_confidence = prom.Histogram(
            "spamshield_prediction_confidence",
            "Prediction confidence distribution",
            ["model", "prediction"],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
        )
        self.active_predictions = prom.Gauge(
            "spamshield_active_predictions",
            "Currently processing predictions",
        )

        # System Metrics
        self.model_load_time = prom.Histogram(
            "spamshield_model_load_seconds",
            "Time to load a model",
            ["model"],
        )

    def record_request(self, method: str, endpoint: str, status: int, duration: float) -> None:
        """Record an HTTP request."""
        if not self._enabled:
            return
        self.http_requests_total.labels(method, endpoint, str(status)).inc()
        self.http_request_duration.labels(method, endpoint).observe(duration)

    def record_prediction(
        self, model: str, prediction: str, confidence: float, latency_ms: float
    ) -> None:
        """Record a prediction event."""
        if not self._enabled:
            return
        self.predictions_total.labels(model, prediction).inc()
        self.prediction_latency.labels(model).observe(latency_ms)
        self.prediction_confidence.labels(model, prediction).observe(confidence)

    def prediction_in_progress(self) -> None:
        if self._enabled:
            self.active_predictions.inc()

    def prediction_complete(self) -> None:
        if self._enabled:
            self.active_predictions.dec()


# Singleton
metrics = MetricsCollector()


def get_metrics_app():
    """Create a WSGI app for /metrics endpoint (mount in FastAPI)."""
    prom = _get_prometheus()
    if prom is None:
        return None
    return prom.make_asgi_app()
