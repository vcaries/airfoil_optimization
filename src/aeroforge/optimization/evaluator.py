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

from aeroforge.core.exceptions import GeneratorError, GeometryError, SolverError
from aeroforge.core.logging import get_logger
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

_log = get_logger(__name__)

# Callable signature: takes a parameter mapping, returns an airfoil.
AirfoilFactory = Callable[[dict[str, float]], "Airfoil"]

# Sentinel values returned when a candidate cannot be evaluated. They are
# finite (pymoo dislikes NaN/inf in objectives) but extreme enough that even
# a hand-tuned candidate would never produce them, so failed designs are
# always dominated.
_FAILED_OBJECTIVE: float = 1.0e6
_FAILED_CONSTRAINT: float = 1.0e3


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

    # ------------------------------------------------------------------ #
    # Shape helpers
    # ------------------------------------------------------------------ #
    @property
    def n_obj(self) -> int:
        """int: Number of objectives."""
        return len(self.objectives)

    @property
    def n_geometric_constraints(self) -> int:
        """int: Number of geometric (cheap) constraints."""
        return len(self.geometric_constraints)

    @property
    def n_physical_constraints(self) -> int:
        """int: Number of physical (solver-dependent) constraints."""
        return len(self.physical_constraints)

    @property
    def n_constr(self) -> int:
        """int: Total constraint count (geometric + physical)."""
        return self.n_geometric_constraints + self.n_physical_constraints

    # ------------------------------------------------------------------ #
    # Convenience: genome -> airfoil
    # ------------------------------------------------------------------ #
    def genome_to_airfoil(self, x: FloatArray) -> Airfoil:
        """Decode a genome vector into the corresponding :class:`Airfoil`.

        This is a thin convenience wrapper around the design space and the
        airfoil factory. It is what visualization helpers call so they can
        reconstruct the best-so-far airfoil for each generation without
        importing the optimization layer themselves.

        Args:
            x: The flat genome vector in :attr:`design_space` order.

        Returns:
            The :class:`Airfoil` produced by :attr:`airfoil_factory`.
        """
        params = self.design_space.to_mapping(x)
        return self.airfoil_factory(params)

    # ------------------------------------------------------------------ #
    # The hot path
    # ------------------------------------------------------------------ #
    def evaluate(self, x: FloatArray) -> tuple[list[float], list[float]]:
        """Evaluate one genome.

        The pipeline is, in order:

        1. Decode the genome to a parameter mapping using the design space.
        2. Build the airfoil through :attr:`airfoil_factory`.
        3. Score geometric constraints (cheap; no solver call required).
        4. If any geometric constraint is violated, short-circuit with
           sentinel objective values — the solver call is expensive, so we
           skip it for candidates we already know are infeasible.
        5. Otherwise, run the solver and score physical constraints
           and objectives from the result.

        Failures (bad genome, generator raises, solver crashes or fails to
        converge) are translated to finite sentinel values rather than
        exceptions, because pymoo's evaluation loop expects every genome to
        produce a numeric ``(F, G)`` tuple.

        Args:
            x: The flat genome vector in :attr:`design_space` order.

        Returns:
            A tuple ``(F, G)`` where ``F`` is a list of length
            :attr:`n_obj` and ``G`` is a list of length :attr:`n_constr`
            (``<= 0`` is feasible).
        """
        n_obj = self.n_obj
        n_geo = self.n_geometric_constraints
        n_phys = self.n_physical_constraints

        # --- 1. Decode genome ------------------------------------------------
        try:
            params = self.design_space.to_mapping(x)
        except ValueError as exc:
            _log.warning("Bad genome decoding: %s", exc)
            return _failed(n_obj, n_geo, n_phys)

        # --- 2. Build airfoil ------------------------------------------------
        try:
            airfoil = self.airfoil_factory(params)
        except (GeneratorError, GeometryError) as exc:
            _log.info("airfoil_factory rejected genome %s: %s", params, exc)
            return _failed(n_obj, n_geo, n_phys)
        except Exception as exc:  # noqa: BLE001 - convert any factory crash
            _log.warning("airfoil_factory crashed on %s: %s", params, exc)
            return _failed(n_obj, n_geo, n_phys)

        # --- 3. Geometric constraints ---------------------------------------
        geo_g = [float(c.evaluate(airfoil)) for c in self.geometric_constraints]
        if any(g > 0.0 for g in geo_g):
            # Skip solver — the candidate is geometrically infeasible. Fill
            # the physical-constraint slots with a positive sentinel so the
            # candidate is dominated on the constraint axis too.
            return (
                [_FAILED_OBJECTIVE] * n_obj,
                geo_g + [_FAILED_CONSTRAINT] * n_phys,
            )

        # --- 4. Solver run --------------------------------------------------
        try:
            result = self.solver.analyze(airfoil, self.operating_point)
        except SolverError as exc:
            _log.info("Solver failed on genome %s: %s", params, exc)
            return (
                [_FAILED_OBJECTIVE] * n_obj,
                geo_g + [_FAILED_CONSTRAINT] * n_phys,
            )

        # --- 5. Score objectives & physical constraints ---------------------
        phys_g = [float(c.evaluate(result)) for c in self.physical_constraints]
        obj_f = [float(o.evaluate(result)) for o in self.objectives]
        return obj_f, geo_g + phys_g


def _failed(n_obj: int, n_geo: int, n_phys: int) -> tuple[list[float], list[float]]:
    """Build the sentinel ``(F, G)`` tuple for a fully failed candidate."""
    return (
        [_FAILED_OBJECTIVE] * n_obj,
        [_FAILED_CONSTRAINT] * (n_geo + n_phys),
    )
