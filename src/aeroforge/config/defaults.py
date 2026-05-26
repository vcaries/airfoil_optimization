"""Library-wide default values.

Centralizing magic numbers here gives a single point of change when tuning
behavior across the library (e.g. raising the default XFOIL iteration cap).
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------- #
# Geometry defaults
# --------------------------------------------------------------------------- #
DEFAULT_N_POINTS_PER_SURFACE: Final[int] = 100
"""Cosine-spaced stations per surface used by generators by default."""

DEFAULT_REPANEL_POINTS: Final[int] = 160
"""Target panel count when repaneling an imported airfoil for XFOIL."""

# --------------------------------------------------------------------------- #
# Solver defaults
# --------------------------------------------------------------------------- #
DEFAULT_XFOIL_BINARY: Final[str] = "xfoil"
"""Executable name searched on PATH unless overridden by settings."""

DEFAULT_MAX_ITER: Final[int] = 200
"""Maximum XFOIL viscous iterations before declaring non-convergence."""

DEFAULT_N_CRIT: Final[float] = 9.0
"""Default transition amplification factor (clean-tunnel value)."""

DEFAULT_PROCESS_TIMEOUT_S: Final[float] = 60.0
"""Hard timeout (seconds) for a single XFOIL invocation."""

# --------------------------------------------------------------------------- #
# Optimization defaults
# --------------------------------------------------------------------------- #
DEFAULT_POPULATION_SIZE: Final[int] = 40
"""Default population size for genetic algorithms."""

DEFAULT_N_GENERATIONS: Final[int] = 50
"""Default number of generations for evolutionary optimization."""

# --------------------------------------------------------------------------- #
# Visualization defaults
# --------------------------------------------------------------------------- #
DEFAULT_FIGURE_DPI: Final[int] = 150
"""Default DPI for saved figures (portfolio-quality without bloat)."""

DEFAULT_ANIMATION_FPS: Final[int] = 12
"""Default frames-per-second for generated GIF/MP4 animations."""
