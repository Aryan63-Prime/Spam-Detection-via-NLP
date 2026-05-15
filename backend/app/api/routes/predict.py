"""
Prediction Routes
===================
Core prediction endpoints for spam classification.

Endpoints:
- POST /api/v1/predict       — Classify a single text
- POST /api/v1/predict/batch — Classify multiple texts
- GET  /api/v1/predict/history — Get prediction history
- GET  /api/v1/predict/models — List available models
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_user, get_optional_user
from backend.app.core.database import get_db
from backend.app.models.models import User
from backend.app.schemas.schemas import (
    APIResponse,
    PredictBatchRequest,
    PredictRequest,
    PredictionHistoryItem,
    PredictionResponse,
)
from backend.app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Predictions"])


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Classify a single text as spam or ham",
)
async def predict(
    payload: PredictRequest,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user),
):
    """
    Classify a text message as spam or ham.

    - Supports multiple ML models (traditional, DL, transformers)
    - Optional explainability via LIME or SHAP
    - Configurable classification threshold
    """
    service = PredictionService(db)

    try:
        result = await service.classify_text(
            text=payload.text,
            model_name=payload.model,
            explain=payload.explain,
            explain_method=payload.explain_method,
            threshold=payload.threshold,
            user_id=user.id if user else None,
            source="api",
        )
        return result

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available: {e}",
        )
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Please try again.",
        )


@router.post(
    "/batch",
    response_model=list[PredictionResponse],
    summary="Classify multiple texts in batch",
)
async def predict_batch(
    payload: PredictBatchRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Classify multiple texts in a single request (max 100).
    Requires authentication.
    """
    if len(payload.texts) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 texts per batch.",
        )

    service = PredictionService(db)

    try:
        results = await service.classify_batch(
            texts=payload.texts,
            model_name=payload.model,
            threshold=payload.threshold,
            user_id=user.id,
        )
        return results

    except Exception as e:
        logger.error("Batch prediction failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed.",
        )


@router.get(
    "/history",
    response_model=list[PredictionHistoryItem],
    summary="Get prediction history",
)
async def get_history(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get the authenticated user's prediction history."""
    service = PredictionService(db)
    return await service.get_history(
        user_id=user.id,
        limit=min(limit, 200),
        offset=offset,
    )


@router.get(
    "/models",
    response_model=APIResponse,
    summary="List available models",
)
async def list_models():
    """List all available trained models for prediction."""
    from backend.app.services.prediction_service import get_inference_engine
    engine = get_inference_engine()
    models = engine.get_available_models()
    return APIResponse(
        data={"models": models, "default": engine.default_model}
    )
