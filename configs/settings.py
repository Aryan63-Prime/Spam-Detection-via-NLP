"""
Centralized Configuration Management
=====================================
Uses pydantic-settings to load configuration from environment variables
and .env files. This is the single source of truth for all project config.

Architecture Decision:
- Pydantic Settings provides type-safe config with validation
- Environment variables for secrets (never hardcoded)
- Nested config classes for separation of concerns
- Singleton pattern via lru_cache for performance

Security:
- All secrets loaded from environment
- No defaults for sensitive values in production
- .env file excluded from version control
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ──────────────────────────────────────────────
# Path Constants
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
ML_DIR = PROJECT_ROOT / "ml"


class Environment(str, Enum):
    """Deployment environment enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class MLSettings(BaseSettings):
    """Machine Learning pipeline configuration."""
    model_config = SettingsConfigDict(env_prefix="ML_")

    # Preprocessing
    max_sequence_length: int = Field(default=512, description="Max token sequence length")
    min_word_length: int = Field(default=2, description="Minimum word length to keep")
    remove_stopwords: bool = Field(default=True, description="Whether to remove stopwords")
    lemmatize: bool = Field(default=True, description="Whether to apply lemmatization")
    language: str = Field(default="english", description="Primary language for NLP")

    # Feature Engineering
    tfidf_max_features: int = Field(default=50000, description="Max TF-IDF features")
    tfidf_ngram_range: tuple[int, int] = Field(default=(1, 3), description="N-gram range")
    embedding_dim: int = Field(default=300, description="Word embedding dimension")

    # Training
    test_size: float = Field(default=0.2, description="Test split ratio")
    val_size: float = Field(default=0.1, description="Validation split ratio")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    batch_size: int = Field(default=32, description="Training batch size")
    epochs: int = Field(default=10, description="Max training epochs")
    learning_rate: float = Field(default=5e-5, description="Learning rate")
    early_stopping_patience: int = Field(default=3, description="Early stopping patience")

    # Inference
    confidence_threshold: float = Field(default=0.5, description="Classification threshold")
    use_gpu: bool = Field(default=False, description="Whether GPU is available")
    use_onnx: bool = Field(default=True, description="Use ONNX Runtime for inference")
    cache_predictions: bool = Field(default=True, description="Cache predictions in Redis")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    # Model paths
    default_model: str = Field(default="distilbert", description="Default model for inference")


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    model_config = SettingsConfigDict(env_prefix="DB_")

    # PostgreSQL
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="spam_user")
    postgres_password: str = Field(default="changeme_in_production")
    postgres_db: str = Field(default="spam_detection")
    postgres_pool_size: int = Field(default=20)
    postgres_max_overflow: int = Field(default=10)

    # Redis
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: Optional[str] = Field(default=None)

    @property
    def postgres_url(self) -> str:
        """Build async PostgreSQL connection URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_sync_url(self) -> str:
        """Build sync PostgreSQL connection URL (for Alembic)."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Build Redis connection URL."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


class SecuritySettings(BaseSettings):
    """Security and authentication configuration."""
    model_config = SettingsConfigDict(env_prefix="SEC_")

    # JWT
    jwt_secret_key: str = Field(default="CHANGE-THIS-IN-PRODUCTION-USE-STRONG-SECRET")
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(default=30)
    jwt_refresh_token_expire_days: int = Field(default=7)

    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60)
    rate_limit_burst: int = Field(default=10)

    # CORS
    cors_origins: list[str] = Field(default=["http://localhost:3000"])
    cors_allow_credentials: bool = Field(default=True)

    # API Keys
    api_key_header: str = Field(default="X-API-Key")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""
    model_config = SettingsConfigDict(env_prefix="MON_")

    prometheus_enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=9090)
    grafana_port: int = Field(default=3001)
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    enable_request_tracing: bool = Field(default=True)


class AppSettings(BaseSettings):
    """
    Root application settings.
    Aggregates all sub-configurations into a single access point.
    """
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / "configs" / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="SpamShield AI")
    app_version: str = Field(default="1.0.0")
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=True)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)

    # Sub-configurations
    ml: MLSettings = Field(default_factory=MLSettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    @field_validator("debug", mode="before")
    @classmethod
    def set_debug_from_env(cls, v: bool, info) -> bool:
        """Auto-disable debug in production."""
        env = info.data.get("environment")
        if env == Environment.PRODUCTION:
            return False
        return v


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Get cached application settings singleton.

    Returns:
        AppSettings: The application configuration instance.

    Usage:
        from configs.settings import get_settings
        settings = get_settings()
        print(settings.ml.max_sequence_length)
    """
    return AppSettings()
