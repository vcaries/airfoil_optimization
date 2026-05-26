"""Chain-of-responsibility pipeline for XFOIL convergence strategies.

A :class:`ConvergencePipeline` is itself a :class:`ConvergenceStrategy`, which
means pipelines can be nested. The pipeline applies its child strategies in
order, returning the first converged result and raising
:class:`ConvergenceError` only when every strategy has failed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aeroforge.core.exceptions import ConvergenceError
from aeroforge.core.logging import get_logger
from aeroforge.core.types import OperatingPoint
from aeroforge.solver.convergence.base import ConvergenceStrategy

if TYPE_CHECKING:
    from aeroforge.geometry.airfoil import Airfoil
    from aeroforge.solver.base import AbstractSolver
    from aeroforge.solver.xfoil.results import PolarPoint

_log = get_logger(__name__)


class ConvergencePipeline(ConvergenceStrategy):
    """Apply a list of fallback strategies in order until one succeeds.

    Args:
        strategies: Ordered list of strategies to attempt.
    """

    def __init__(self, strategies: list[ConvergenceStrategy]) -> None:
        """Store the ordered strategy list."""
        if not strategies:
            raise ValueError("ConvergencePipeline requires at least one strategy.")
        self.strategies = list(strategies)

    @property
    def name(self) -> str:
        """str: A pipeline label including the child-strategy names."""
        return "Pipeline[" + " > ".join(s.name for s in self.strategies) + "]"

    def attempt(
        self,
        solver: AbstractSolver,
        airfoil: Airfoil,
        point: OperatingPoint,
        *,
        history: list[PolarPoint],
    ) -> PolarPoint:
        """Try each strategy in turn, returning the first success.

        Args:
            solver: The underlying solver.
            airfoil: The airfoil being analyzed.
            point: The operating point that needs help converging.
            history: Previously converged points (passed through to children).

        Returns:
            The :class:`PolarPoint` returned by the first successful strategy.

        Raises:
            ConvergenceError: If every strategy raises.
        """
        last_error: ConvergenceError | None = None
        for strategy in self.strategies:
            try:
                _log.debug("Trying %s for alpha=%.3f deg", strategy.name, point.alpha)
                return strategy.attempt(solver, airfoil, point, history=history)
            except ConvergenceError as err:
                _log.info(
                    "%s failed at alpha=%.3f deg: %s",
                    strategy.name,
                    point.alpha,
                    err,
                )
                last_error = err
        assert last_error is not None
        raise ConvergenceError(
            f"All {len(self.strategies)} strategies failed for alpha={point.alpha}.",
            alpha=point.alpha,
        ) from last_error
