"""pymoo :class:`Problem` adapter for aeroforge.

Connects the framework-agnostic :class:`AirfoilEvaluator` to pymoo's
:class:`Problem` API so any pymoo algorithm (NSGA-II, NSGA-III, CMA-ES,
single-objective GA) can drive an aeroforge optimization without any aeroforge
code changes.

Requires the ``optim`` extra (``pip install aeroforge[optim]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

# pymoo is an optional dependency; this module is part of the ``optim`` extra.
from pymoo.core.problem import Problem  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from aeroforge.optimization.evaluator import AirfoilEvaluator


class AirfoilProblem(Problem):
    """A pymoo :class:`Problem` whose evaluations route through an evaluator.

    Args:
        evaluator: The :class:`AirfoilEvaluator` doing the real work.
    """

    def __init__(self, evaluator: AirfoilEvaluator) -> None:
        """Wire the design space and counts into the pymoo base class."""
        lower, upper = evaluator.design_space.bounds
        n_obj = evaluator.n_obj
        n_ieq_constr = evaluator.n_constr
        super().__init__(
            n_var=evaluator.design_space.n_var,
            n_obj=n_obj,
            n_ieq_constr=n_ieq_constr,
            xl=lower,
            xu=upper,
        )
        self._evaluator = evaluator

    def _evaluate(
        self,
        x: np.ndarray,
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Pymoo evaluation hook.

        Args:
            x: ``(pop_size, n_var)`` matrix of genome vectors.
            out: Output dict; pymoo expects ``out["F"]`` (and optionally
                ``out["G"]``).
            *args: Forwarded positional args from pymoo (unused here).
            **kwargs: Forwarded keyword args from pymoo (unused here).
        """
        pop = x.shape[0]
        n_obj = self.n_obj
        n_constr = self.n_ieq_constr

        f_out = np.empty((pop, n_obj), dtype=np.float64)
        g_out = np.empty((pop, n_constr), dtype=np.float64) if n_constr > 0 else None

        for i in range(pop):
            f, g = self._evaluator.evaluate(x[i])
            f_out[i, :] = f
            if g_out is not None:
                g_out[i, :] = g

        out["F"] = f_out
        if g_out is not None:
            out["G"] = g_out
