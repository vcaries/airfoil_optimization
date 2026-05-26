"""Objective-function abstractions.

An :class:`Objective` maps a converged :class:`PolarPoint` (or set of points)
to a scalar that pymoo will minimise. The :func:`minimize_drag`,
:func:`maximize_lift_to_drag`, etc. factories return ready-to-use instances.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aeroforge.solver.xfoil.results import PolarPoint


class Objective(ABC):
    """Map a converged result to a scalar minimisation target.

    pymoo minimises by convention; objectives that should be maximised return
    the negated quantity.
    """

    @property
    def name(self) -> str:
        """str: A short label used in logs, history, and plots."""
        return type(self).__name__

    @abstractmethod
    def evaluate(self, result: PolarPoint) -> float:
        """Compute the scalar objective from ``result``.

        Args:
            result: A converged :class:`PolarPoint`.

        Returns:
            The scalar value to minimise.
        """
        raise NotImplementedError


@dataclass(slots=True)
class MinimizeDrag(Objective):
    """Minimise ``C_d`` at the analysed operating point."""

    def evaluate(self, result: PolarPoint) -> float:
        """Return ``C_d``."""
        return float(result.cd)


@dataclass(slots=True)
class MaximizeLift(Objective):
    """Maximise ``C_l`` (returns ``-C_l``)."""

    def evaluate(self, result: PolarPoint) -> float:
        """Return ``-C_l`` so pymoo minimises it."""
        return float(-result.cl)


@dataclass(slots=True)
class MaximizeLiftToDrag(Objective):
    """Maximise ``L/D`` (returns ``-C_l/C_d``)."""

    def evaluate(self, result: PolarPoint) -> float:
        """Return ``-C_l / C_d`` so pymoo minimises it."""
        return float(-result.lift_to_drag)
