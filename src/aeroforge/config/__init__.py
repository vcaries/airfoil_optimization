"""Centralized configuration and library-wide defaults.

This subpackage is intentionally **not** imported by :mod:`aeroforge` at top
level so that ``import aeroforge`` works in minimal environments that do not
have :mod:`pydantic` installed. Import :mod:`aeroforge.config` explicitly when
you need :func:`get_settings`.
"""

from aeroforge.config import defaults
from aeroforge.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings", "defaults"]
