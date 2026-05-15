"""
Authentication Routes
=======================
Handles user registration, login, token refresh, and profile.

Endpoints:
- POST /api/v1/auth/register — Create new account
- POST /api/v1/auth/login — Get JWT tokens
- POST /api/v1/auth/refresh — Refresh access token
- GET  /api/v1/auth/me — Get current user profile
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_user
from backend.app.core.database import get_db
from backend.app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from backend.app.models.models import User
from backend.app.schemas.schemas import (
    APIResponse,
    TokenRefresh,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=APIResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(
    payload: UserRegister,
    db: AsyncSession = Depends(get_db),
):
    """Create a new user account with email and password."""

    # Check existing email
    result = await db.execute(
        select(User).where(User.email == payload.email)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered.",
        )

    # Check existing username
    result = await db.execute(
        select(User).where(User.username == payload.username)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already taken.",
        )

    # Create user
    user = User(
        email=payload.email,
        username=payload.username,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        role="user",
    )
    db.add(user)
    await db.flush()

    logger.info("User registered: %s (%s)", payload.email, user.id)

    return APIResponse(
        success=True,
        message="Account created successfully.",
        data=UserResponse.model_validate(user).model_dump(),
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and get JWT tokens",
)
async def login(
    payload: UserLogin,
    db: AsyncSession = Depends(get_db),
):
    """Authenticate with email and password. Returns access + refresh tokens."""

    # Find user
    result = await db.execute(
        select(User).where(User.email == payload.email)
    )
    user = result.scalar_one_or_none()

    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated.",
        )

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    await db.flush()

    # Generate tokens
    access_token = create_access_token(str(user.id), user.role)
    refresh_token = create_refresh_token(str(user.id), user.role)

    logger.info("User logged in: %s", user.email)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=1800,  # 30 minutes
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
)
async def refresh_token(
    payload: TokenRefresh,
    db: AsyncSession = Depends(get_db),
):
    """Get a new access token using a refresh token."""

    try:
        token_data = decode_token(payload.refresh_token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token.",
        )

    if token_data.token_type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not a refresh token.",
        )

    # Generate new access token
    access_token = create_access_token(token_data.sub, token_data.role)

    return TokenResponse(
        access_token=access_token,
        refresh_token=payload.refresh_token,
        expires_in=1800,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
async def get_profile(
    user: User = Depends(get_current_user),
):
    """Returns the authenticated user's profile."""
    return UserResponse.model_validate(user)
