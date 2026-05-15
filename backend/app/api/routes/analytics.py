"""
Analytics Routes
==================
Dashboard analytics and model benchmark endpoints.

Endpoints:
- GET /api/v1/analytics/summary — Dashboard summary stats
- GET /api/v1/analytics/benchmarks — Model benchmark results
- GET /api/v1/analytics/timeline — Prediction timeline data
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_user
from backend.app.core.database import get_db
from backend.app.models.models import Prediction, User
from backend.app.schemas.schemas import AnalyticsSummary, APIResponse, ModelBenchmark
from backend.app.services.prediction_service import PredictionService
from ml.config import MODELS_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get(
    "/summary",
    response_model=AnalyticsSummary,
    summary="Get dashboard analytics summary",
)
async def get_summary(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Returns aggregate statistics for the analytics dashboard:
    total predictions, spam ratio, average confidence, etc.
    """
    service = PredictionService(db)
    return await service.get_analytics()


@router.get(
    "/benchmarks",
    response_model=list[ModelBenchmark],
    summary="Get model benchmark results",
)
async def get_benchmarks():
    """Return the latest model benchmark comparison results."""
    benchmark_path = MODELS_DIR / "benchmark_traditional.json"

    if not benchmark_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No benchmark results found. Run training first.",
        )

    with open(benchmark_path) as f:
        data = json.load(f)

    return [
        ModelBenchmark(
            model_name=item["model_name"],
            accuracy=item["accuracy"],
            precision=item["precision"],
            recall=item["recall"],
            f1=item["f1"],
            roc_auc=item["roc_auc"],
        )
        for item in data
    ]


@router.get(
    "/timeline",
    response_model=APIResponse,
    summary="Get prediction timeline data",
)
async def get_timeline(
    days: int = Query(7, ge=1, le=90, description="Number of days to look back."),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Returns daily prediction counts for the timeline chart.
    Grouped by day and prediction label.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Query daily counts grouped by prediction
    result = await db.execute(
        select(
            func.date(Prediction.created_at).label("date"),
            Prediction.prediction,
            func.count(Prediction.id).label("count"),
        )
        .where(Prediction.created_at >= cutoff)
        .group_by(func.date(Prediction.created_at), Prediction.prediction)
        .order_by(func.date(Prediction.created_at))
    )

    rows = result.all()

    # Build timeline data
    timeline = {}
    for row in rows:
        date_str = str(row.date)
        if date_str not in timeline:
            timeline[date_str] = {"date": date_str, "spam": 0, "ham": 0, "total": 0}
        timeline[date_str][row.prediction] = row.count
        timeline[date_str]["total"] += row.count

    return APIResponse(data=list(timeline.values()))
