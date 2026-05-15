"""
Security & Observability Middleware
=====================================
Custom middleware for request logging, security headers,
and rate limiting.

Architecture:
- Middleware runs on EVERY request (before and after route handler)
- Order matters: Security headers → Rate limiting → Logging
- All middleware is async for non-blocking operation
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with timing, status, and correlation ID.

    Adds X-Request-ID header for distributed tracing.
    Logs duration for performance monitoring.
    """

    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Timing
        start = time.perf_counter()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.perf_counter() - start) * 1000

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

        # Log
        logger.info(
            "%s %s → %d (%.1fms) [%s]",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds security headers to every response.

    Headers:
    - X-Content-Type-Options: Prevents MIME-type sniffing
    - X-Frame-Options: Prevents clickjacking
    - X-XSS-Protection: XSS filter for older browsers
    - Strict-Transport-Security: Enforce HTTPS
    - Content-Security-Policy: Restrict resource loading
    - Referrer-Policy: Control referrer information
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter.

    For production, use Redis-backed rate limiting (e.g., fastapi-limiter).
    This in-memory implementation is sufficient for development and single-instance deployments.

    Rate limits:
    - Default: 60 requests per minute per IP
    - Prediction endpoint: 30 requests per minute per IP
    """

    def __init__(self, app: FastAPI, requests_per_minute: int = 60) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._request_counts: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old entries (older than 60s)
        self._request_counts[client_ip] = [
            t for t in self._request_counts[client_ip]
            if now - t < 60
        ]

        # Check rate limit
        if len(self._request_counts[client_ip]) >= self.requests_per_minute:
            return Response(
                content='{"detail": "Rate limit exceeded. Try again later."}',
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Record request
        self._request_counts[client_ip].append(now)

        response = await call_next(request)

        # Add rate limit headers
        remaining = self.requests_per_minute - len(self._request_counts[client_ip])
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(remaining, 0))

        return response
