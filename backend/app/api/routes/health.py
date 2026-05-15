"""
Health Check Routes
=====================
Kubernetes-compatible health and readiness probes.

Endpoints:
- GET /health — Liveness probe (app is running)
- GET /ready — Readiness probe (DB + models loaded)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, status
from sqlalchemy import text

from backend.app.schemas.schemas import APIResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=APIResponse,
    summary="Liveness probe",
)
async def health():
    """Returns 200 if the application is running."""
    return APIResponse(
        message="SpamShield AI is healthy.",
        data={
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
        },
    )


@router.get(
    "/ready",
    response_model=APIResponse,
    summary="Readiness probe",
)
async def readiness():
    """
    Returns 200 if the application is ready to serve requests.
    Checks database connectivity and model availability.
    """
    checks = {
        "database": False,
        "ml_engine": False,
    }

    # Check database
    try:
        from backend.app.core.database import get_engine
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as e:
        logger.warning("DB readiness check failed: %s", e)

    # Check ML engine
    try:
        from backend.app.services.prediction_service import get_inference_engine
        engine = get_inference_engine()
        models = engine.get_available_models()
        checks["ml_engine"] = len(models) > 0
    except Exception as e:
        logger.warning("ML readiness check failed: %s", e)

    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503

    return APIResponse(
        success=all_ready,
        message="Ready" if all_ready else "Not ready",
        data={"checks": checks},
    )
