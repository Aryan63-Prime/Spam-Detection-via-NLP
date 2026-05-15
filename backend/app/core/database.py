"""
Database Module
================
Async SQLAlchemy engine and session management for PostgreSQL.

Architecture:
- Async engine with connection pooling
- Scoped sessions via async context manager
- Dependency injection for FastAPI routes
- Base model class for ORM inheritance

Performance:
- Connection pooling (pool_size=20, max_overflow=10)
- Async driver (asyncpg) for non-blocking I/O
- Prepared statements via SQLAlchemy
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from backend.app.core import get_app_settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Naming Convention (Alembic-friendly)
# ──────────────────────────────────────────────
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    metadata = MetaData(naming_convention=convention)


# ──────────────────────────────────────────────
# Engine & Session Factory
# ──────────────────────────────────────────────

_engine = None
_session_factory = None


def get_engine():
    """Get or create the async SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_app_settings()
        _engine = create_async_engine(
            settings.db.postgres_url,
            pool_size=settings.db.postgres_pool_size,
            max_overflow=settings.db.postgres_max_overflow,
            pool_pre_ping=True,
            echo=settings.debug,
        )
        logger.info("Database engine created: %s", settings.db.postgres_host)
    return _engine


def get_session_factory():
    """Get or create the async session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency: provides an async database session.

    Usage in routes:
        @router.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            ...
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables (for development only)."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created.")


async def close_db() -> None:
    """Close the database engine."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("Database engine closed.")
