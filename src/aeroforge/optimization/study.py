"""High-level optimization facade.

:class:`OptimizationStudy` bundles the evaluator, algorithm, and termination
criteria into one object so users can launch a run in two lines:

.. code-block:: python

    study = OptimizationStudy(evaluator, algorithm=nsga2(pop_size=40), n_gen=50)
    result = study.run()

The study is also responsible for resuming an interrupted optimization from a
:class:`HistoryCallback` checkpoint.

Requires the ``optim`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aeroforge.optimization.callbacks import HistoryCallback

if TYPE_CHECKING:
    from aeroforge.optimization.evaluator import AirfoilEvaluator

PathLike = str | Path


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
    """

    evaluator: AirfoilEvaluator
    algorithm: Any
    n_gen: int = 50
    seed: int = 1
    history: HistoryCallback = field(default_factory=HistoryCallback)
    checkpoint_path: PathLike | None = None

    def run(self) -> Any:
        """Execute the optimization.

        Returns:
            The pymoo :class:`Result` object.

        Raises:
            NotImplementedError: Implementation planned for milestone M3.
        """
        raise NotImplementedError("OptimizationStudy.run (planned, M3).")

    def resume(self) -> Any:
        """Resume an interrupted study from :attr:`checkpoint_path`.

        Returns:
            The pymoo :class:`Result` object.

        Raises:
            NotImplementedError: Implementation planned for milestone M3.
        """
        raise NotImplementedError("OptimizationStudy.resume (planned, M3).")
