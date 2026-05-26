"""Shared types, enums, and lightweight value objects used across aeroforge.

Keeping these in one dependency-light module (only NumPy) avoids circular
imports between the geometry, solver, and optimization layers, all of which
need to speak a common vocabulary (arrays, surfaces, operating points).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

#: Canonical floating-point array type used throughout the library.
FloatArray = npt.NDArray[np.float64]


class Surface(str, Enum):
    """The two sides of an airfoil contour."""

    UPPER = "upper"
    LOWER = "lower"


class TrailingEdge(str, Enum):
    """Trailing-edge closure convention for generated airfoils.

    Attributes:
        OPEN: The classic NACA polynomial, leaving a small finite TE gap.
        CLOSED: Modified last coefficient so the TE thickness is exactly zero,
            which is friendlier for paneling and meshing.
    """

    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """An aerodynamic operating point for a viscous XFOIL analysis.

    Attributes:
        alpha: Angle of attack in degrees.
        reynolds: Chord-based Reynolds number. A value of ``0`` requests an
            inviscid analysis.
        mach: Free-stream Mach number (compressibility correction).
        n_crit: Transition amplification factor (e^N method). Typical values
            range from 9 (clean wind tunnel) down to ~4 (rough/noisy flow).
        x_trip_upper: Forced transition location on the upper surface (x/c),
            or ``None`` for free transition.
        x_trip_lower: Forced transition location on the lower surface (x/c),
            or ``None`` for free transition.
    """

    alpha: float
    reynolds: float = 1.0e6
    mach: float = 0.0
    n_crit: float = 9.0
    x_trip_upper: float | None = None
    x_trip_lower: float | None = None

    @property
    def is_viscous(self) -> bool:
        """bool: Whether this operating point requests a viscous solution."""
        return self.reynolds > 0.0
