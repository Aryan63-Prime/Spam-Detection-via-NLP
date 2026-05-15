"""
Feedback Routes
=================
User feedback endpoints for prediction corrections.

Endpoints:
- POST /api/v1/feedback — Submit feedback on a prediction
- GET  /api/v1/feedback — Get user's submitted feedback
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_user
from backend.app.core.database import get_db
from backend.app.models.models import Feedback, Prediction, User
from backend.app.schemas.schemas import APIResponse, FeedbackCreate, FeedbackResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit prediction feedback",
)
async def submit_feedback(
    payload: FeedbackCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Submit a correction for a prediction.
    Used for collecting retraining data when the model is wrong.
    """
    # Verify prediction exists
    result = await db.execute(
        select(Prediction).where(Prediction.id == payload.prediction_id)
    )
    prediction = result.scalar_one_or_none()
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found.",
        )

    # Check for existing feedback
    existing = await db.execute(
        select(Feedback).where(Feedback.prediction_id == payload.prediction_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Feedback already submitted for this prediction.",
        )

    # Create feedback
    feedback = Feedback(
        prediction_id=payload.prediction_id,
        user_id=user.id,
        correct_label=payload.correct_label,
        comment=payload.comment,
    )
    db.add(feedback)
    await db.flush()

    logger.info(
        "Feedback submitted: prediction=%s, correct_label=%s",
        payload.prediction_id, payload.correct_label,
    )

    return FeedbackResponse.model_validate(feedback)


@router.get(
    "",
    response_model=list[FeedbackResponse],
    summary="Get submitted feedback",
)
async def get_feedback(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get the user's submitted feedback history."""
    result = await db.execute(
        select(Feedback)
        .where(Feedback.user_id == user.id)
        .order_by(Feedback.created_at.desc())
        .limit(min(limit, 200))
        .offset(offset)
    )
    items = result.scalars().all()
    return [FeedbackResponse.model_validate(f) for f in items]
