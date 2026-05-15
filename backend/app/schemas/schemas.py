"""
Pydantic Schemas (DTOs)
========================
Request/response validation schemas for all API endpoints.

Design:
- Strict input validation (Pydantic V2)
- Separate Create/Update/Response schemas (no leaky abstractions)
- Consistent response envelope pattern
- Proper field constraints and examples for OpenAPI docs
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# ──────────────────────────────────────────────
# Base Response Envelope
# ──────────────────────────────────────────────

class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool = True
    message: str = "OK"
    data: Any = None
    errors: Optional[list[str]] = None


# ──────────────────────────────────────────────
# Auth Schemas
# ──────────────────────────────────────────────

class UserRegister(BaseModel):
    """User registration request."""
    email: EmailStr = Field(..., example="user@example.com")
    username: str = Field(..., min_length=3, max_length=50, example="john_doe")
    password: str = Field(..., min_length=8, max_length=128, example="SecurePass123!")
    full_name: Optional[str] = Field(None, max_length=255, example="John Doe")

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit.")
        return v


class UserLogin(BaseModel):
    """User login request."""
    email: EmailStr = Field(..., example="user@example.com")
    password: str = Field(..., example="SecurePass123!")


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Access token expiry in seconds")


class TokenRefresh(BaseModel):
    """Token refresh request."""
    refresh_token: str


class UserResponse(BaseModel):
    """User profile response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    username: str
    full_name: Optional[str]
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime]


# ──────────────────────────────────────────────
# Prediction Schemas
# ──────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Single prediction request."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        example="Congratulations! You've won a $1000 gift card. Click here to claim.",
    )
    model: Optional[str] = Field(
        None,
        example="logistic_regression",
        description="Model to use. Options: naive_bayes, logistic_regression, svm, xgboost, distilbert",
    )
    explain: bool = Field(False, description="Include XAI explanation.")
    explain_method: str = Field("lime", description="Explanation method: lime or shap.")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Classification threshold.")


class PredictBatchRequest(BaseModel):
    """Batch prediction request."""
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to classify (max 100).",
    )
    model: Optional[str] = Field(None)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Single prediction response."""
    model_config = ConfigDict(from_attributes=True)

    text: str
    prediction: str
    confidence: float
    spam_probability: float
    ham_probability: float
    model_used: str
    processing_time_ms: float
    metadata: Optional[dict] = None
    explanation: Optional[dict] = None


class PredictionHistoryItem(BaseModel):
    """Prediction history list item."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    input_text: str
    prediction: str
    confidence: float
    model_used: str
    created_at: datetime


# ──────────────────────────────────────────────
# Feedback Schemas
# ──────────────────────────────────────────────

class FeedbackCreate(BaseModel):
    """Feedback submission request."""
    prediction_id: uuid.UUID = Field(..., description="ID of the prediction to correct.")
    correct_label: str = Field(
        ...,
        pattern="^(spam|ham)$",
        description="The correct label: 'spam' or 'ham'.",
    )
    comment: Optional[str] = Field(None, max_length=1000)


class FeedbackResponse(BaseModel):
    """Feedback submission response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    prediction_id: uuid.UUID
    correct_label: str
    comment: Optional[str]
    created_at: datetime


# ──────────────────────────────────────────────
# Analytics Schemas
# ──────────────────────────────────────────────

class AnalyticsSummary(BaseModel):
    """Dashboard analytics summary."""
    total_predictions: int
    total_spam: int
    total_ham: int
    spam_ratio: float
    avg_confidence: float
    avg_processing_time_ms: float
    predictions_today: int
    models_available: list[str]


class ModelBenchmark(BaseModel):
    """Model benchmark result."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
