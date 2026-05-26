"""aeroforge -- a professional Python toolkit for 2D airfoil aerodynamics.

The top-level namespace re-exports the most commonly used building blocks
(:class:`Airfoil`, generators, exceptions). Heavier subpackages with optional
dependencies (:mod:`aeroforge.optimization` pulls in pymoo,
:mod:`aeroforge.visualization` pulls in matplotlib) are intentionally **not**
imported here, so ``import aeroforge`` stays cheap and works in stripped-down
environments such as CI runners or HPC nodes.

Example:
    >>> import aeroforge
    >>> airfoil = aeroforge.NACA4Generator("2412").generate()
    >>> airfoil.name
    'NACA 2412'
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from aeroforge.core import (
    AeroforgeError,
    ConfigurationError,
    ConvergenceError,
    GeneratorError,
    GeometryError,
    InvalidAirfoilError,
    OperatingPoint,
    OptimizationError,
    ParsingError,
    SolverError,
    Surface,
    TrailingEdge,
    XfoilExecutionError,
    XfoilNotFoundError,
    configure_logging,
    get_logger,
)
from aeroforge.geometry import (
    Airfoil,
    AirfoilGenerator,
    BezierGenerator,
    CSTGenerator,
    DatFileGenerator,
    NACA4Generator,
    PARSECGenerator,
)

try:
    __version__ = version("aeroforge")
except PackageNotFoundError:  # pragma: no cover - editable install fallback
    __version__ = "0.0.0+local"

__all__ = [
    "__version__",
    # geometry
    "Airfoil",
    "AirfoilGenerator",
    "NACA4Generator",
    "DatFileGenerator",
    "CSTGenerator",
    "BezierGenerator",
    "PARSECGenerator",
    # core types
    "Surface",
    "TrailingEdge",
    "OperatingPoint",
    # logging
    "get_logger",
    "configure_logging",
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
]
