"""
Prediction Service
====================
Business logic for spam classification predictions.

Architecture:
- Service layer separates business logic from API routes
- Coordinates between ML inference engine and database
- Handles prediction caching, history storage, and analytics
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.models import Prediction
from backend.app.schemas.schemas import (
    AnalyticsSummary,
    PredictionHistoryItem,
    PredictionResponse,
)
from ml.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)

# Global inference engine (loaded once at startup)
_inference_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Get or create the global inference engine."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngine(default_model="logistic_regression")
    return _inference_engine


class PredictionService:
    """
    Business logic for prediction operations.

    Responsibilities:
    - Run ML inference
    - Store prediction history in DB
    - Retrieve prediction history
    - Generate analytics summaries
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.engine = get_inference_engine()

    async def classify_text(
        self,
        text: str,
        model_name: Optional[str] = None,
        explain: bool = False,
        explain_method: str = "lime",
        threshold: float = 0.5,
        user_id: Optional[UUID] = None,
        source: str = "api",
    ) -> PredictionResponse:
        """
        Classify a single text and store the result.

        Args:
            text: Raw input text.
            model_name: Model to use (or default).
            explain: Whether to include XAI explanation.
            explain_method: "lime" or "shap".
            threshold: Classification threshold.
            user_id: Authenticated user ID (optional).
            source: Request source (api, web, batch).

        Returns:
            PredictionResponse with classification result.
        """
        # Run inference
        result = self.engine.predict(
            text=text,
            model_name=model_name,
            explain=explain,
            explain_method=explain_method,
            threshold=threshold,
        )

        # Store in database
        prediction = Prediction(
            user_id=user_id,
            input_text=text,
            prediction=result.prediction,
            confidence=result.confidence,
            spam_probability=result.spam_probability,
            ham_probability=result.ham_probability,
            model_used=result.model_used,
            processing_time_ms=result.processing_time_ms,
            has_url=result.metadata.get("has_url", False),
            text_length=result.metadata.get("original_length", len(text)),
            spam_keyword_count=result.metadata.get("spam_keyword_count", 0),
            source=source,
        )
        self.db.add(prediction)
        await self.db.flush()

        return PredictionResponse(
            text=text,
            prediction=result.prediction,
            confidence=result.confidence,
            spam_probability=result.spam_probability,
            ham_probability=result.ham_probability,
            model_used=result.model_used,
            processing_time_ms=result.processing_time_ms,
            metadata=result.metadata,
            explanation=result.explanation,
        )

    async def classify_batch(
        self,
        texts: list[str],
        model_name: Optional[str] = None,
        threshold: float = 0.5,
        user_id: Optional[UUID] = None,
    ) -> list[PredictionResponse]:
        """Classify a batch of texts."""
        results = self.engine.predict_batch(
            texts=texts,
            model_name=model_name,
            threshold=threshold,
        )

        # Store all in DB
        for result in results:
            prediction = Prediction(
                user_id=user_id,
                input_text=result.text,
                prediction=result.prediction,
                confidence=result.confidence,
                spam_probability=result.spam_probability,
                ham_probability=result.ham_probability,
                model_used=result.model_used,
                processing_time_ms=result.processing_time_ms,
                source="batch",
            )
            self.db.add(prediction)

        await self.db.flush()

        return [
            PredictionResponse(
                text=r.text,
                prediction=r.prediction,
                confidence=r.confidence,
                spam_probability=r.spam_probability,
                ham_probability=r.ham_probability,
                model_used=r.model_used,
                processing_time_ms=r.processing_time_ms,
            )
            for r in results
        ]

    async def get_history(
        self,
        user_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PredictionHistoryItem]:
        """Get prediction history for a user."""
        query = (
            select(Prediction)
            .where(Prediction.user_id == user_id)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.db.execute(query)
        predictions = result.scalars().all()

        return [
            PredictionHistoryItem.model_validate(p)
            for p in predictions
        ]

    async def get_analytics(self) -> AnalyticsSummary:
        """Generate analytics summary for the dashboard."""
        # Total predictions
        total_q = await self.db.execute(select(func.count(Prediction.id)))
        total = total_q.scalar() or 0

        # Spam count
        spam_q = await self.db.execute(
            select(func.count(Prediction.id)).where(Prediction.prediction == "spam")
        )
        total_spam = spam_q.scalar() or 0
        total_ham = total - total_spam

        # Average confidence
        avg_conf_q = await self.db.execute(
            select(func.avg(Prediction.confidence))
        )
        avg_confidence = float(avg_conf_q.scalar() or 0)

        # Average processing time
        avg_time_q = await self.db.execute(
            select(func.avg(Prediction.processing_time_ms))
        )
        avg_time = float(avg_time_q.scalar() or 0)

        # Today's predictions
        from datetime import datetime, timezone
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        today_q = await self.db.execute(
            select(func.count(Prediction.id)).where(Prediction.created_at >= today_start)
        )
        predictions_today = today_q.scalar() or 0

        # Available models
        available_models = self.engine.get_available_models()

        return AnalyticsSummary(
            total_predictions=total,
            total_spam=total_spam,
            total_ham=total_ham,
            spam_ratio=total_spam / max(total, 1),
            avg_confidence=avg_confidence,
            avg_processing_time_ms=avg_time,
            predictions_today=predictions_today,
            models_available=available_models,
        )
