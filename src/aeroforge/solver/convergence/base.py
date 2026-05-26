"""Abstract base class for XFOIL convergence-helper strategies.

Each strategy embodies one technique known to improve XFOIL convergence:
ramping iterations, sweeping alpha from a converged neighbour, switching
between viscous and inviscid initialisation, etc. Strategies are composable
via :class:`~aeroforge.solver.convergence.pipeline.ConvergencePipeline`, which
implements a chain-of-responsibility over a list of strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from aeroforge.core.types import OperatingPoint

if TYPE_CHECKING:
    from aeroforge.geometry.airfoil import Airfoil
    from aeroforge.solver.base import AbstractSolver
    from aeroforge.solver.xfoil.results import PolarPoint


class ConvergenceStrategy(ABC):
    """One technique for coaxing a stubborn XFOIL run into converging.

    A strategy is *stateless across runs*: it receives the failing context and
    a solver handle and either returns a converged :class:`PolarPoint` or
    raises :class:`ConvergenceError` so the next strategy in the pipeline can
    try its luck.
    """

    @property
    def name(self) -> str:
        """str: A short human-readable strategy name for logging."""
        return type(self).__name__

    @abstractmethod
    def attempt(
        self,
        solver: AbstractSolver,
        airfoil: Airfoil,
        point: OperatingPoint,
        *,
        history: list[PolarPoint],
    ) -> PolarPoint:
        """Try to converge ``point`` using this strategy.

        Args:
            solver: The underlying solver to invoke.
            airfoil: The airfoil being analyzed.
            point: The operating point that failed under the default settings.
            history: Previously converged points in the current sweep, ordered
                in evaluation order. Some strategies (e.g. alpha continuation)
                use the closest converged neighbour as a warm-start hint.

        Returns:
            A converged :class:`PolarPoint`.

        Raises:
            ConvergenceError: If this strategy also fails to converge.
        """
        raise NotImplementedError
