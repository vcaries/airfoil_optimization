"""High-level optimization facade.

:class:`OptimizationStudy` bundles the evaluator, algorithm, and termination
criteria into one object so users can launch a run in two lines:

.. code-block:: python

    study = OptimizationStudy(evaluator, algorithm=nsga2(pop_size=40), n_gen=50)
    result = study.run()

The study is also responsible for checkpointing the :class:`HistoryCallback`
so a study can be resumed from where it left off (with the caveat that
pymoo's internal algorithm state is not picklable: ``resume`` continues from
the last seed-deterministic start; only the captured history is preserved).

Requires the ``optim`` extra.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pymoo.optimize import minimize  # type: ignore[import-not-found]

from aeroforge.core.exceptions import OptimizationError
from aeroforge.core.logging import get_logger
from aeroforge.optimization.callbacks import HistoryCallback
from aeroforge.optimization.problem import AirfoilProblem

if TYPE_CHECKING:
    from aeroforge.optimization.evaluator import AirfoilEvaluator

PathLike = str | Path
_log = get_logger(__name__)


@dataclass(slots=True)
class OptimizationStudy:
    """High-level driver of one optimization run.

    Attributes:
        evaluator: The :class:`AirfoilEvaluator` that translates genomes to
            objective and constraint values.
        algorithm: A configured pymoo algorithm (e.g. from
            :mod:`aeroforge.optimization.algorithms`).
        n_gen: Number of generations.
        seed: PRNG seed for reproducibility.
        history: Callback accumulating per-generation snapshots.
        checkpoint_path: Optional path to persist the history between runs.
        verbose: Whether to let pymoo print its own progress.
    """

    evaluator: AirfoilEvaluator
    algorithm: Any
    n_gen: int = 50
    seed: int = 1
    history: HistoryCallback = field(default_factory=HistoryCallback)
    checkpoint_path: PathLike | None = None
    verbose: bool = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> Any:
        """Execute the optimization.

        Wires the :class:`AirfoilProblem`, the pymoo algorithm, the
        :class:`HistoryCallback`, and the termination criterion together,
        then dispatches to :func:`pymoo.optimize.minimize`.

        Returns:
            The pymoo :class:`Result` object. ``result.X`` holds the best
            genome, ``result.F`` its objective value(s), ``result.G`` its
            constraint value(s), and ``self.history`` is populated with one
            :class:`GenerationSnapshot` per generation.

        Raises:
            OptimizationError: If pymoo itself raises during the run.
        """
        problem = AirfoilProblem(self.evaluator)
        _log.info(
            "Running optimization: pop=%s n_gen=%d seed=%d",
            getattr(self.algorithm, "pop_size", "?"),
            self.n_gen,
            self.seed,
        )
        try:
            result = minimize(
                problem,
                self.algorithm,
                ("n_gen", self.n_gen),
                seed=self.seed,
                callback=self.history,
                verbose=self.verbose,
                save_history=False,
            )
        except Exception as exc:  # noqa: BLE001 - reframe under typed exception
            raise OptimizationError(f"pymoo run failed: {exc}") from exc

        if self.checkpoint_path is not None:
            self.save_checkpoint(self.checkpoint_path)
        _log.info("Optimization finished; %d generation(s) captured", len(self.history))
        return result

    def resume(self) -> Any:
        """Resume an interrupted study from :attr:`checkpoint_path`.

        Note:
            pymoo's algorithm state is not reliably picklable, so a true
            warm-restart is out of scope for v0.1. This method reloads the
            history snapshots from disk (so previous progress is not lost
            for visualization purposes) and then re-launches a fresh run
            from the same seed, which is deterministic.

        Returns:
            The pymoo :class:`Result` object from the new run.

        Raises:
            OptimizationError: If no checkpoint is configured.
        """
        if self.checkpoint_path is None:
            raise OptimizationError("OptimizationStudy.resume requires a checkpoint_path.")
        self.load_checkpoint(self.checkpoint_path)
        return self.run()

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: PathLike) -> Path:
        """Pickle the :class:`HistoryCallback` snapshots to ``path``.

        Args:
            path: Destination file. Parent directories are created on demand.

        Returns:
            The path that was written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.history.snapshots, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _log.debug("Saved %d snapshots to %s", len(self.history.snapshots), path)
        return path

    def load_checkpoint(self, path: PathLike) -> None:
        """Replace :attr:`history`'s snapshots with those read from ``path``.

        Args:
            path: Source file written by :meth:`save_checkpoint`.

        Raises:
            OptimizationError: If the checkpoint file is missing.
        """
        path = Path(path)
        if not path.exists():
            raise OptimizationError(f"Checkpoint not found: {path}")
        with path.open("rb") as handle:
            self.history.snapshots = pickle.load(handle)
        _log.info("Loaded %d snapshots from %s", len(self.history.snapshots), path)
