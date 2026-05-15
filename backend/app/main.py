"""
SpamShield AI — FastAPI Application Entry Point
=================================================
Production-grade async REST API for spam detection.

Architecture:
- FastAPI with async endpoints
- JWT authentication + RBAC
- Rate limiting + security headers
- OpenAPI auto-documentation
- Kubernetes health probes
- CORS for frontend communication

Run:
    uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.api.routes import auth, predict, feedback, analytics, health
from backend.app.core.database import close_db, init_db
from backend.app.core.logging_config import setup_logging
from backend.app.middleware.middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Application Lifespan
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.

    Startup:
    - Initialize logging
    - Create database tables
    - Pre-load ML models

    Shutdown:
    - Close database connections
    - Release ML model memory
    """
    # ── Startup ──
    setup_logging(level="INFO", log_format="text")
    logger.info("=" * 60)
    logger.info("SpamShield AI — Starting up...")
    logger.info("=" * 60)

    # Initialize database
    try:
        await init_db()
        logger.info("✓ Database initialized.")
    except Exception as e:
        logger.warning("Database init skipped (will retry on first request): %s", e)

    # Pre-load ML inference engine
    try:
        from backend.app.services.prediction_service import get_inference_engine
        engine = get_inference_engine()
        models = engine.get_available_models()
        logger.info("✓ ML engine initialized. Available models: %s", models)
    except Exception as e:
        logger.warning("ML engine init deferred: %s", e)

    logger.info("SpamShield AI — Ready to serve requests.")
    logger.info("=" * 60)

    yield

    # ── Shutdown ──
    logger.info("SpamShield AI — Shutting down...")
    await close_db()
    logger.info("SpamShield AI — Shutdown complete.")


# ──────────────────────────────────────────────
# Application Factory
# ──────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Application factory pattern.
    Creates and configures the FastAPI application.
    """
    app = FastAPI(
        title="SpamShield AI",
        description=(
            "Industry-grade AI-powered spam detection REST API. "
            "Supports traditional ML, deep learning, and transformer models "
            "with explainable AI and real-time analytics."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",    # Next.js dev
            "http://localhost:3001",    # Next.js alt
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    )

    # ── Middleware (order matters: last added = first executed) ──
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=120)

    # ── API Routes ───────────────────────────────
    API_V1 = "/api/v1"

    app.include_router(health.router)
    app.include_router(auth.router, prefix=API_V1)
    app.include_router(predict.router, prefix=API_V1)
    app.include_router(feedback.router, prefix=API_V1)
    app.include_router(analytics.router, prefix=API_V1)

    # ── Root endpoint ────────────────────────────
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": "SpamShield AI",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Create the app instance
app = create_app()
