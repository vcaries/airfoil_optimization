"""Genome -> Airfoil -> Solver -> Metrics translation.

The :class:`AirfoilEvaluator` is the bridge between pymoo (which speaks
``numpy`` genome vectors) and the rest of aeroforge (which speaks airfoils and
operating points). Keeping it in its own module makes optimization problems
straightforward to assemble: pick a generator, a list of objectives and
constraints, an operating point, and the evaluator handles the wiring.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aeroforge.core.types import FloatArray, OperatingPoint
from aeroforge.optimization.variables import DesignSpace

if TYPE_CHECKING:
    from aeroforge.geometry.airfoil import Airfoil
    from aeroforge.optimization.constraints import (
        GeometricConstraint,
        PhysicalConstraint,
    )
    from aeroforge.optimization.objectives import Objective
    from aeroforge.solver.base import AbstractSolver


# Callable signature: takes a parameter mapping, returns an airfoil.
AirfoilFactory = Callable[[dict[str, float]], "Airfoil"]


@dataclass(slots=True)
class AirfoilEvaluator:
    """Evaluate one design genome end to end.

    Attributes:
        design_space: The set of design variables.
        airfoil_factory: Callable mapping a ``{name: value}`` parameter dict
            to a fresh :class:`Airfoil` instance.
        solver: Backend aerodynamic solver.
        operating_point: The operating point at which to evaluate.
        objectives: Ordered list of objectives.
        geometric_constraints: Geometric constraints (cheap; evaluated first).
        physical_constraints: Aerodynamic constraints (require a solver call).
    """

    design_space: DesignSpace
    airfoil_factory: AirfoilFactory
    solver: AbstractSolver
    operating_point: OperatingPoint
    objectives: list[Objective] = field(default_factory=list)
    geometric_constraints: list[GeometricConstraint] = field(default_factory=list)
    physical_constraints: list[PhysicalConstraint] = field(default_factory=list)

    def evaluate(self, x: FloatArray) -> tuple[list[float], list[float]]:
        """Evaluate one genome.

        Args:
            x: The flat genome vector in :attr:`design_space` order.

        Returns:
            A tuple ``(F, G)`` where ``F`` is the list of objective values and
            ``G`` is the list of constraint values (``<= 0`` is feasible).

        Raises:
            EvaluationError: If the airfoil cannot be built or the solver
                cannot converge after all fallbacks.
            NotImplementedError: Implementation planned for milestone M3.
        """
        raise NotImplementedError("AirfoilEvaluator.evaluate (planned, M3).")
