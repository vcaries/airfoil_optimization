"""Runtime configuration, loaded from environment variables or ``.env`` files.

Settings are managed by :mod:`pydantic_settings`, which gives us validation,
type coercion, and 12-factor-style configuration for free. Resolution order
(later wins): defaults < environment variables < explicit constructor args.

All environment variables share the ``AEROFORGE_`` prefix, e.g.::

    export AEROFORGE_XFOIL_BINARY=/opt/xfoil/bin/xfoil
    export AEROFORGE_MAX_ITER=400
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from aeroforge.config import defaults


class Settings(BaseSettings):
    """Validated, immutable runtime settings.

    Attributes:
        xfoil_binary: Path or executable name for the XFOIL binary.
        work_dir: Working directory where solver scratch files are written.
        max_iter: Default viscous iteration cap.
        n_crit: Default transition amplification factor.
        process_timeout_s: Hard timeout per XFOIL invocation, in seconds.
        log_level: Logging level for the library root logger.
    """

    model_config = SettingsConfigDict(
        env_prefix="AEROFORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        frozen=True,
    )

    xfoil_binary: str = Field(
        default=defaults.DEFAULT_XFOIL_BINARY,
        description="Path or PATH-resolvable name of the XFOIL executable.",
    )
    work_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "runs",
        description="Directory used for XFOIL scratch files and run artifacts.",
    )
    max_iter: int = Field(default=defaults.DEFAULT_MAX_ITER, ge=1)
    n_crit: float = Field(default=defaults.DEFAULT_N_CRIT, gt=0.0)
    process_timeout_s: float = Field(default=defaults.DEFAULT_PROCESS_TIMEOUT_S, gt=0.0)
    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton.

    The result is cached for the lifetime of the process; call
    :meth:`functools.lru_cache.cache_clear` on this function in tests to reset.

    Returns:
        The resolved :class:`Settings` instance.
    """
    return Settings()
