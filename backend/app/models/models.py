"""
ORM Models
===========
SQLAlchemy ORM models for Users, Predictions, Feedback.

Database Design:
- users: Authentication and RBAC
- predictions: Prediction history with confidence scores
- feedback: User corrections for retraining pipeline

Indexing Strategy:
- users.email: Unique index for login lookups
- predictions.user_id: FK index for user history
- predictions.created_at: Index for time-range queries
- feedback.prediction_id: FK index for feedback lookups
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.database import Base


def utc_now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


class User(Base):
    """
    User account model.

    Supports:
    - Email/password authentication
    - Role-based access control (user, admin)
    - Account activation/deactivation
    - API key for programmatic access
    """
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    username: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    api_key: Mapped[Optional[str]] = mapped_column(String(64), unique=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"User(id={self.id}, email={self.email}, role={self.role})"


class Prediction(Base):
    """
    Prediction history model.

    Stores every prediction for:
    - Analytics dashboards
    - User history
    - Retraining data collection
    - Audit trail
    """
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    prediction: Mapped[str] = mapped_column(String(10), nullable=False)  # "spam" / "ham"
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    spam_probability: Mapped[float] = mapped_column(Float, nullable=False)
    ham_probability: Mapped[float] = mapped_column(Float, nullable=False)
    model_used: Mapped[str] = mapped_column(String(50), nullable=False)
    processing_time_ms: Mapped[float] = mapped_column(Float, default=0.0)

    # Metadata
    has_url: Mapped[bool] = mapped_column(Boolean, default=False)
    text_length: Mapped[int] = mapped_column(Integer, default=0)
    spam_keyword_count: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(20), default="api")  # api, web, batch

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )

    # Relationships
    user: Mapped[Optional["User"]] = relationship(back_populates="predictions")
    feedback: Mapped[Optional["Feedback"]] = relationship(
        back_populates="prediction", uselist=False
    )

    # Indexes for analytics queries
    __table_args__ = (
        Index("ix_predictions_user_created", "user_id", "created_at"),
        Index("ix_predictions_prediction", "prediction"),
    )

    def __repr__(self) -> str:
        return f"Prediction(id={self.id}, prediction={self.prediction}, confidence={self.confidence})"


class Feedback(Base):
    """
    User feedback model for prediction corrections.

    Critical for continuous learning:
    - Users can mark false positives/negatives
    - Collected feedback feeds the retraining pipeline
    - Tracks feedback patterns for model improvement
    """
    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False, unique=True
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    correct_label: Mapped[str] = mapped_column(String(10), nullable=False)  # "spam" / "ham"
    comment: Mapped[Optional[str]] = mapped_column(Text)
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    # Relationships
    prediction: Mapped["Prediction"] = relationship(back_populates="feedback")

    def __repr__(self) -> str:
        return f"Feedback(id={self.id}, correct_label={self.correct_label})"
