"""Batch execution of computational campaigns.

The :class:`CampaignRunner` is the orchestrator that drives the solver across a
:class:`Sweep`, in parallel if asked, with optional retry through a
:class:`ConvergencePipeline` and result persistence via a :class:`ResultStore`.

Process-level parallelism is preferred over threads because XFOIL is CPU-bound
and the wrapper launches one subprocess per evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aeroforge.campaigns.store import ResultStore
    from aeroforge.campaigns.sweep import Sweep
    from aeroforge.solver.base import AbstractSolver
    from aeroforge.solver.convergence.base import ConvergenceStrategy


@dataclass(slots=True)
class CampaignRunner:
    """Orchestrate the execution of a sweep through a solver.

    Args:
        solver: Backend solver (typically :class:`XfoilRunner`).
        convergence: Optional fallback strategy for non-converged points.
        store: Optional result store for resumable / persistent campaigns.
        n_workers: Number of worker processes. ``1`` runs serially.
    """

    solver: AbstractSolver
    convergence: ConvergenceStrategy | None = None
    store: ResultStore | None = None
    n_workers: int = 1

    def run(self, sweep: Sweep) -> None:
        """Execute ``sweep`` to completion.

        Args:
            sweep: The :class:`Sweep` describing the work to do.

        Raises:
            NotImplementedError: Implementation planned for milestone M4.
        """
        raise NotImplementedError("CampaignRunner.run (planned, M4).")
