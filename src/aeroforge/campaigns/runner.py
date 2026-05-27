"""Batch execution of computational campaigns.

The :class:`CampaignRunner` is the orchestrator that drives the solver across a
:class:`Sweep`, in parallel if asked, with optional retry through a
:class:`ConvergencePipeline` and result persistence via a :class:`ResultStore`.

Process-level parallelism is preferred over threads because XFOIL is CPU-bound
and the wrapper launches one subprocess per evaluation.

Resumability
------------
When a :class:`ResultStore` is wired in, the runner consults it before every
solver call. A re-run of the same sweep against the same store therefore
skips everything that previously converged -- ``Ctrl+C`` and re-launching is
a first-class workflow.

Parallelism caveats
-------------------
* Convergence strategies that warm-start from converged neighbours (e.g.
  :class:`AlphaContinuationStrategy`) only have access to an *empty* history
  in worker processes, so they degrade to a no-op there. Prefer serial mode
  if alpha-continuation is important.
* The solver and convergence-pipeline objects are pickled to workers, so
  they must be picklable. Subprocess-based solvers (:class:`XfoilRunner`)
  are fine; lambdas in custom strategies are not.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import TYPE_CHECKING

from aeroforge.campaigns.store import hash_airfoil
from aeroforge.core.exceptions import ConvergenceError, SolverError
from aeroforge.core.logging import get_logger

if TYPE_CHECKING:
    from aeroforge.campaigns.store import ResultStore
    from aeroforge.campaigns.sweep import Sweep
    from aeroforge.core.types import OperatingPoint
    from aeroforge.geometry.airfoil import Airfoil
    from aeroforge.solver.base import AbstractSolver
    from aeroforge.solver.convergence.base import ConvergenceStrategy
    from aeroforge.solver.xfoil.results import PolarPoint

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Public result container
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class CampaignResult:
    """The outcome of a campaign run.

    Attributes:
        converged: All converged :class:`PolarPoint` results, in the order
            they were produced (serial mode) or completed (parallel mode).
        failed: Operating points (paired with their airfoil hash) for which
            neither the solver nor the convergence pipeline could converge.
    """

    converged: list[PolarPoint] = field(default_factory=list)
    failed: list[tuple[str, OperatingPoint]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """float: Fraction of points that converged (0.0 -- 1.0)."""
        total = len(self.converged) + len(self.failed)
        return len(self.converged) / total if total else 0.0


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
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

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, sweep: Sweep | Iterable[tuple[Airfoil, OperatingPoint]]) -> CampaignResult:
        """Execute ``sweep`` to completion.

        Args:
            sweep: A :class:`Sweep` or any iterable yielding
                ``(airfoil, operating_point)`` tuples.

        Returns:
            A :class:`CampaignResult` summarising what converged and what
            failed.
        """
        items = list(sweep)
        _log.info(
            "Campaign starting: %d work units, n_workers=%d, store=%s, convergence=%s",
            len(items),
            self.n_workers,
            type(self.store).__name__ if self.store else None,
            type(self.convergence).__name__ if self.convergence else None,
        )
        if self.n_workers <= 1:
            return self._run_serial(items)
        return self._run_parallel(items)

    # ------------------------------------------------------------------ #
    # Serial path
    # ------------------------------------------------------------------ #
    def _run_serial(self, items: list[tuple[Airfoil, OperatingPoint]]) -> CampaignResult:
        """Execute the work units one by one in the current process.

        Serial mode is the only path where convergence strategies that
        consult the live history (e.g. alpha continuation) work end to end.
        """
        result = CampaignResult()
        for airfoil, point in items:
            af_hash = hash_airfoil(airfoil)

            # 1. Skip if already in the store.
            if self.store is not None and self.store.has(af_hash, point):
                cached = self.store.load(af_hash, point)
                if cached is not None:
                    result.converged.append(cached)
                    continue

            # 2. Run the solver, with optional convergence fallback.
            pt = self._try_one(airfoil, point, history=result.converged)
            if pt is None:
                result.failed.append((af_hash, point))
                continue

            # 3. Persist successful results.
            if self.store is not None:
                self.store.write(af_hash, pt)
            result.converged.append(pt)

        _log.info(
            "Campaign finished: %d/%d converged (%.0f%%)",
            len(result.converged),
            len(items),
            100.0 * result.success_rate,
        )
        return result

    def _try_one(
        self,
        airfoil: Airfoil,
        point: OperatingPoint,
        *,
        history: list[PolarPoint],
    ) -> PolarPoint | None:
        """Solve a single point, attempting the convergence pipeline on failure."""
        try:
            return self.solver.analyze(airfoil, point)
        except ConvergenceError:
            if self.convergence is None:
                return None
            try:
                return self.convergence.attempt(self.solver, airfoil, point, history=history)
            except ConvergenceError:
                return None
        except SolverError:
            # Process-level failures (timeout / crash) are also "soft" failures
            # at the campaign level: we record and move on.
            return None

    # ------------------------------------------------------------------ #
    # Parallel path
    # ------------------------------------------------------------------ #
    def _run_parallel(self, items: list[tuple[Airfoil, OperatingPoint]]) -> CampaignResult:
        """Execute the work units across a multiprocessing pool.

        The parent process owns the :class:`ResultStore`, so we never need
        to make it picklable. Workers receive ``(solver, convergence,
        airfoil, point)`` tuples and return ``(airfoil_hash, result_or_none)``.
        """
        # Build the work items, skipping anything the store already has.
        units: list[tuple[AbstractSolver, ConvergenceStrategy | None, Airfoil, OperatingPoint]] = []
        cached: list[PolarPoint] = []
        for airfoil, point in items:
            af_hash = hash_airfoil(airfoil)
            if self.store is not None and self.store.has(af_hash, point):
                cached_pt = self.store.load(af_hash, point)
                if cached_pt is not None:
                    cached.append(cached_pt)
                    continue
            units.append((self.solver, self.convergence, airfoil, point))

        result = CampaignResult(converged=list(cached))
        if not units:
            _log.info("Every point was already cached; nothing to dispatch.")
            return result

        with Pool(processes=self.n_workers) as pool:
            for af_hash, point, pt in pool.imap_unordered(_worker_evaluate, units):
                if pt is None:
                    result.failed.append((af_hash, point))
                    continue
                if self.store is not None:
                    self.store.write(af_hash, pt)
                result.converged.append(pt)

        _log.info(
            "Parallel campaign finished: %d/%d converged (%.0f%%)",
            len(result.converged),
            len(items),
            100.0 * result.success_rate,
        )
        return result


# --------------------------------------------------------------------------- #
# Worker (module-level, picklable)
# --------------------------------------------------------------------------- #
def _worker_evaluate(
    unit: tuple[AbstractSolver, ConvergenceStrategy | None, Airfoil, OperatingPoint],
) -> tuple[str, OperatingPoint, PolarPoint | None]:
    """Evaluate a single work unit. Module-level so :mod:`multiprocessing` can pickle it.

    Returns the original :class:`OperatingPoint` alongside the result so the
    parent process can attribute failures without re-matching by hash.
    """
    solver, convergence, airfoil, point = unit
    af_hash = hash_airfoil(airfoil)
    try:
        return af_hash, point, solver.analyze(airfoil, point)
    except ConvergenceError:
        if convergence is None:
            return af_hash, point, None
        try:
            return af_hash, point, convergence.attempt(solver, airfoil, point, history=[])
        except ConvergenceError:
            return af_hash, point, None
    except SolverError:
        return af_hash, point, None
