"""
Backend Core Configuration
============================
Loads settings from environment variables via pydantic-settings.
Single source of truth for all backend configuration.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

# Add project root to path for ML imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import AppSettings, get_settings


@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    """Cached application settings accessor for the backend."""
    return get_settings()
