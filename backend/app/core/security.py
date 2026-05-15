"""
Security Module
================
JWT authentication, password hashing, and token management.

Architecture:
- JWT access + refresh token pattern
- Bcrypt password hashing (slow by design, brute-force resistant)
- Token blacklisting support via Redis
- Dependency injection for FastAPI route protection

Security Decisions:
- HS256 for JWT (symmetric, fast, sufficient for single-service)
- 30-min access tokens, 7-day refresh tokens
- Bcrypt with 12 rounds (industry standard)
- No secrets in tokens — only user ID and role
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from backend.app.core import get_app_settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Password Hashing
# ──────────────────────────────────────────────

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,
)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


# ──────────────────────────────────────────────
# JWT Token Management
# ──────────────────────────────────────────────

class TokenPayload(BaseModel):
    """JWT token payload schema."""
    sub: str  # User ID
    role: str = "user"
    exp: Optional[datetime] = None
    token_type: str = "access"


def create_access_token(
    user_id: str,
    role: str = "user",
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        user_id: Unique user identifier.
        role: User role (user, admin, etc.).
        expires_delta: Custom expiration. Defaults to config value.

    Returns:
        Encoded JWT string.
    """
    settings = get_app_settings()
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.security.jwt_access_token_expire_minutes)

    expire = datetime.now(timezone.utc) + expires_delta

    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": expire,
        "token_type": "access",
        "iat": datetime.now(timezone.utc),
    }

    token = jwt.encode(
        payload,
        settings.security.jwt_secret_key,
        algorithm=settings.security.jwt_algorithm,
    )
    return token


def create_refresh_token(user_id: str, role: str = "user") -> str:
    """Create a JWT refresh token with longer expiry."""
    settings = get_app_settings()
    expires_delta = timedelta(days=settings.security.jwt_refresh_token_expire_days)
    expire = datetime.now(timezone.utc) + expires_delta

    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": expire,
        "token_type": "refresh",
        "iat": datetime.now(timezone.utc),
    }

    return jwt.encode(
        payload,
        settings.security.jwt_secret_key,
        algorithm=settings.security.jwt_algorithm,
    )


def decode_token(token: str) -> TokenPayload:
    """
    Decode and validate a JWT token.

    Raises:
        JWTError: If token is invalid or expired.
    """
    settings = get_app_settings()
    try:
        payload = jwt.decode(
            token,
            settings.security.jwt_secret_key,
            algorithms=[settings.security.jwt_algorithm],
        )
        return TokenPayload(**payload)
    except JWTError as e:
        logger.warning("JWT decode failed: %s", e)
        raise
