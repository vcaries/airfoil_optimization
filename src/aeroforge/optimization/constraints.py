"""Constraint abstractions.

Constraints follow pymoo's "<= 0 is feasible" convention. Each
:class:`Constraint` exposes :meth:`evaluate` returning a real value whose
non-positivity defines feasibility.

Two flavours live here:

* Geometric constraints, evaluated from the :class:`Airfoil` alone.
* Physical (aerodynamic) constraints, evaluated from a converged result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aeroforge.geometry.airfoil import Airfoil
    from aeroforge.solver.xfoil.results import PolarPoint


class GeometricConstraint(ABC):
    """A constraint on the geometry alone."""

    @property
    def name(self) -> str:
        """str: Short label for logging."""
        return type(self).__name__

    @abstractmethod
    def evaluate(self, airfoil: Airfoil) -> float:
        """Return ``g(airfoil)``; feasible if ``g <= 0``."""
        raise NotImplementedError


class PhysicalConstraint(ABC):
    """A constraint on the aerodynamic result."""

    @property
    def name(self) -> str:
        """str: Short label for logging."""
        return type(self).__name__

    @abstractmethod
    def evaluate(self, result: PolarPoint) -> float:
        """Return ``g(result)``; feasible if ``g <= 0``."""
        raise NotImplementedError


@dataclass(slots=True)
class MinThicknessConstraint(GeometricConstraint):
    """Require the maximum thickness to be at least ``t_min``.

    Attributes:
        t_min: Minimum allowed thickness, as a fraction of chord.
    """

    t_min: float

    def evaluate(self, airfoil: Airfoil) -> float:
        """Return ``t_min - max_thickness``."""
        return float(self.t_min - airfoil.max_thickness)


@dataclass(slots=True)
class MinPitchingMomentConstraint(PhysicalConstraint):
    """Require ``C_m`` to be at least ``cm_min`` (e.g. ``-0.1``).

    Attributes:
        cm_min: Minimum allowed pitching-moment coefficient.
    """

    cm_min: float

    def evaluate(self, result: PolarPoint) -> float:
        """Return ``cm_min - C_m``."""
        return float(self.cm_min - result.cm)
