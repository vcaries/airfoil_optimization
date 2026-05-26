"""Abstract solver interface.

Every concrete solver (XFOIL today; potentially MSES, SU2, or a surrogate
model tomorrow) implements :class:`AbstractSolver`. Higher layers (campaigns,
optimization) program against this interface alone, never against a concrete
backend, which keeps backends swappable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from aeroforge.core.types import OperatingPoint

if TYPE_CHECKING:
    from aeroforge.geometry.airfoil import Airfoil
    from aeroforge.solver.xfoil.results import PolarPoint


class AbstractSolver(ABC):
    """Interface for a 2D airfoil aerodynamic solver.

    Concrete solvers may be stateful (XFOIL holds a loaded airfoil) but the
    public API is point-wise: each :meth:`analyze` call is self-contained.
    """

    @abstractmethod
    def analyze(self, airfoil: Airfoil, point: OperatingPoint) -> PolarPoint:
        """Solve a single operating point.

        Args:
            airfoil: The airfoil geometry to analyze.
            point: The aerodynamic operating point.

        Returns:
            A :class:`PolarPoint` with C_l, C_d, C_m, and transition info.

        Raises:
            ConvergenceError: If the solver fails to converge.
            SolverError: For other backend failures.
        """
        raise NotImplementedError
