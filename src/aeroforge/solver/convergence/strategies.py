"""Concrete convergence-helper strategies.

Each class encodes one well-known XFOIL convergence trick. They share the
:class:`ConvergenceStrategy` interface so they can be composed in any order by
the :class:`ConvergencePipeline`.

Strategy reference (planned implementations):

* :class:`IncreaseIterationsStrategy` -- raise ``ITER`` and retry.
* :class:`AlphaContinuationStrategy` -- step alpha in small increments from
  the nearest converged neighbour.
* :class:`InviscidInitStrategy` -- run inviscidly first, then switch to
  viscous to obtain a better starting BL state.
* :class:`RepanelStrategy` -- apply ``PANE`` to fix degenerate panels before
  retrying.
* :class:`PerturbAlphaStrategy` -- nudge alpha by a tiny epsilon (useful near
  the stall corner where XFOIL likes to oscillate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aeroforge.core.types import OperatingPoint
from aeroforge.solver.convergence.base import ConvergenceStrategy

if TYPE_CHECKING:
    from aeroforge.geometry.airfoil import Airfoil
    from aeroforge.solver.base import AbstractSolver
    from aeroforge.solver.xfoil.results import PolarPoint


class IncreaseIterationsStrategy(ConvergenceStrategy):
    """Raise XFOIL's ``ITER`` cap and retry.

    Args:
        factor: Multiplicative factor applied to the current iteration cap.
        max_iter: Absolute ceiling beyond which the strategy gives up.
    """

    def __init__(self, factor: float = 2.0, max_iter: int = 800) -> None:
        """Store the iteration-bump parameters."""
        self.factor = float(factor)
        self.max_iter = int(max_iter)

    def attempt(
        self,
        solver: AbstractSolver,
        airfoil: Airfoil,
        point: OperatingPoint,
        *,
        history: list[PolarPoint],
    ) -> PolarPoint:
        """Retry with a higher iteration cap (planned)."""
        raise NotImplementedError("IncreaseIterationsStrategy (planned, M2).")


class AlphaContinuationStrategy(ConvergenceStrategy):
    """Walk alpha in small steps from the nearest converged neighbour.

    Args:
        step: Alpha increment (degrees) used during continuation.
        max_steps: Maximum number of intermediate alphas before giving up.
    """

    def __init__(self, step: float = 0.25, max_steps: int = 20) -> None:
        """Store the continuation parameters."""
        self.step = float(step)
        self.max_steps = int(max_steps)

    def attempt(
        self,
        solver: AbstractSolver,
        airfoil: Airfoil,
        point: OperatingPoint,
        *,
        history: list[PolarPoint],
    ) -> PolarPoint:
        """Step alpha from the closest converged point (planned)."""
        raise NotImplementedError("AlphaContinuationStrategy (planned, M2).")


class InviscidInitStrategy(ConvergenceStrategy):
    """Run inviscidly first, then switch to viscous with the same alpha.

    Provides a better-conditioned initial BL state, which often unblocks
    near-stall operating points.
    """

    def attempt(
        self,
        solver: AbstractSolver,
        airfoil: Airfoil,
        point: OperatingPoint,
        *,
        history: list[PolarPoint],
    ) -> PolarPoint:
        """Two-stage inviscid -> viscous run (planned)."""
        raise NotImplementedError("InviscidInitStrategy (planned, M2).")


class RepanelStrategy(ConvergenceStrategy):
    """Apply XFOIL's ``PANE`` automatic repaneling and retry.

    Useful when the airfoil arrives with degenerate panels (e.g. from a
    user-supplied ``.dat`` file with awkward spacing).
    """

    def attempt(
        self,
        solver: AbstractSolver,
        airfoil: Airfoil,
        point: OperatingPoint,
        *,
        history: list[PolarPoint],
    ) -> PolarPoint:
        """Run ``PANE`` and retry (planned)."""
        raise NotImplementedError("RepanelStrategy (planned, M2).")


class PerturbAlphaStrategy(ConvergenceStrategy):
    """Nudge alpha by a small epsilon to escape a limit-cycle oscillation.

    Args:
        epsilon: Magnitude of the perturbation, in degrees.
    """

    def __init__(self, epsilon: float = 0.05) -> None:
        """Store the perturbation magnitude."""
        self.epsilon = float(epsilon)

    def attempt(
        self,
        solver: AbstractSolver,
        airfoil: Airfoil,
        point: OperatingPoint,
        *,
        history: list[PolarPoint],
    ) -> PolarPoint:
        """Retry at ``alpha +/- epsilon`` (planned)."""
        raise NotImplementedError("PerturbAlphaStrategy (planned, M2).")
