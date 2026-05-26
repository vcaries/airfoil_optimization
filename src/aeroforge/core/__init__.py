"""Cross-cutting foundations: exceptions, logging, and shared types.

This subpackage has no dependencies beyond NumPy and the standard library so it
can be safely imported from every other layer without risking circular imports.
"""

from aeroforge.core.exceptions import (
    AeroforgeError,
    ConfigurationError,
    ConvergenceError,
    EvaluationError,
    GeneratorError,
    GeometryError,
    InvalidAirfoilError,
    OptimizationError,
    ParsingError,
    SolverError,
    XfoilExecutionError,
    XfoilNotFoundError,
)
from aeroforge.core.logging import configure_logging, get_logger
from aeroforge.core.types import FloatArray, OperatingPoint, Surface, TrailingEdge

__all__ = [
    # exceptions
    "AeroforgeError",
    "ConfigurationError",
    "GeometryError",
    "InvalidAirfoilError",
    "GeneratorError",
    "SolverError",
    "XfoilNotFoundError",
    "XfoilExecutionError",
    "ConvergenceError",
    "ParsingError",
    "OptimizationError",
    "EvaluationError",
    # logging
    "get_logger",
    "configure_logging",
    # types
    "FloatArray",
    "Surface",
    "TrailingEdge",
    "OperatingPoint",
]
