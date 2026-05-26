"""Exception hierarchy for :mod:`aeroforge`.

All library-specific errors derive from :class:`AeroforgeError`. This lets a
caller catch the whole family with a single ``except AeroforgeError`` while
still being able to react to a specific failure mode (e.g. retry only on
:class:`ConvergenceError`).

The hierarchy mirrors the layered architecture::

    AeroforgeError
    ├── ConfigurationError
    ├── GeometryError
    │   ├── InvalidAirfoilError
    │   └── GeneratorError
    ├── SolverError
    │   ├── XfoilNotFoundError
    │   ├── XfoilExecutionError
    │   ├── ConvergenceError
    │   └── ParsingError
    └── OptimizationError
        └── EvaluationError
"""

from __future__ import annotations


class AeroforgeError(Exception):
    """Base class for every exception raised by aeroforge."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class ConfigurationError(AeroforgeError):
    """Raised when configuration values are missing or inconsistent."""


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
class GeometryError(AeroforgeError):
    """Base class for geometry-related failures."""


class InvalidAirfoilError(GeometryError):
    """Raised when airfoil coordinates are malformed or non-physical.

    Examples include too few points, NaN/inf values, mismatched array
    lengths, or a coordinate set that does not describe a closed contour.
    """


class GeneratorError(GeometryError):
    """Raised when an airfoil generator receives invalid parameters."""


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class SolverError(AeroforgeError):
    """Base class for solver (XFOIL) failures."""


class XfoilNotFoundError(SolverError):
    """Raised when the XFOIL executable cannot be located on the system."""


class XfoilExecutionError(SolverError):
    """Raised when the XFOIL process fails, crashes, or times out."""


class ConvergenceError(SolverError):
    """Raised when a viscous solution fails to converge.

    Carries the offending operating point so convergence strategies can decide
    how to adapt (reduce step, ramp iterations, restart, etc.).
    """

    def __init__(self, message: str, *, alpha: float | None = None) -> None:
        """Initialize the error.

        Args:
            message: Human-readable description of the failure.
            alpha: Angle of attack (degrees) at which convergence failed,
                if applicable.
        """
        super().__init__(message)
        self.alpha = alpha


class ParsingError(SolverError):
    """Raised when XFOIL output cannot be parsed into structured results."""


# --------------------------------------------------------------------------- #
# Optimization
# --------------------------------------------------------------------------- #
class OptimizationError(AeroforgeError):
    """Base class for optimization-layer failures."""


class EvaluationError(OptimizationError):
    """Raised when a design candidate cannot be evaluated end to end."""
