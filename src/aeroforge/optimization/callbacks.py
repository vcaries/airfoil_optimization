"""pymoo callbacks for capturing optimization history.

Used by the visualization layer to render generation-by-generation animations
of design evolution, Pareto-front progression, and metric histories.

Requires the ``optim`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pymoo.core.callback import Callback  # type: ignore[import-not-found]

from aeroforge.core.types import FloatArray


@dataclass(slots=True)
class GenerationSnapshot:
    """One generation's state, kept light enough to store many of them.

    Attributes:
        generation: Zero-based generation index.
        x: ``(pop, n_var)`` genome matrix.
        f: ``(pop, n_obj)`` objective matrix.
        g: ``(pop, n_constr)`` constraint matrix (``None`` if no constraints).
    """

    generation: int
    x: FloatArray
    f: FloatArray
    g: FloatArray | None = None


class HistoryCallback(Callback):
    """Capture a :class:`GenerationSnapshot` after each generation.

    The collected history feeds the animation pipeline in
    :mod:`aeroforge.visualization.animation`, the convergence-history plots,
    and the optimization-study checkpoint mechanism.
    """

    def __init__(self) -> None:
        """Initialize pymoo's Callback state plus the snapshot buffer."""
        super().__init__()
        self.snapshots: list[GenerationSnapshot] = []

    def notify(self, algorithm: Any, **kwargs: Any) -> None:
        """Hook called by pymoo once per generation.

        Args:
            algorithm: The driving pymoo algorithm instance.
            **kwargs: Unused, kept for pymoo signature compatibility.
        """
        pop = algorithm.pop
        x_raw = pop.get("X")
        f_raw = pop.get("F")
        g_raw = pop.get("G")
        snapshot = GenerationSnapshot(
            generation=int(algorithm.n_gen) - 1,
            x=np.asarray(x_raw, dtype=float),
            f=np.asarray(f_raw, dtype=float),
            g=np.asarray(g_raw, dtype=float) if g_raw is not None else None,
        )
        self.snapshots.append(snapshot)

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def best_per_generation(self) -> FloatArray:
        """Return the best (minimum) objective value seen at each generation.

        Single-objective only; multi-objective callers should use a Pareto
        plot instead.

        Returns:
            A 1D array of length ``len(snapshots)`` with the minimum F per
            generation, or an empty array if no generations have been
            captured yet.
        """
        if not self.snapshots:
            return np.asarray([], dtype=float)
        return np.asarray([float(np.min(s.f)) for s in self.snapshots], dtype=float)

    def __len__(self) -> int:
        """int: Number of generations captured so far."""
        return len(self.snapshots)
