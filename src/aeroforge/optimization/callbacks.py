"""pymoo callbacks for capturing optimization history.

Used by the visualization layer to render generation-by-generation animations
of design evolution, Pareto-front progression, and metric histories.

Requires the ``optim`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
        g: ``(pop, n_constr)`` constraint matrix.
    """

    generation: int
    x: FloatArray
    f: FloatArray
    g: FloatArray | None = None


@dataclass(slots=True)
class HistoryCallback(Callback):
    """Capture a :class:`GenerationSnapshot` after each generation.

    The collected history feeds the animation pipeline in
    :mod:`aeroforge.visualization.animation`.
    """

    snapshots: list[GenerationSnapshot] = field(default_factory=list)

    def notify(self, algorithm: Any, **kwargs: Any) -> None:  # pragma: no cover
        """Pymoo hook called once per generation.

        Args:
            algorithm: The driving pymoo algorithm instance.
            **kwargs: Unused, kept for pymoo signature compatibility.
        """
        pop = algorithm.pop
        snapshot = GenerationSnapshot(
            generation=algorithm.n_gen - 1,
            x=np.array(pop.get("X"), dtype=float),
            f=np.array(pop.get("F"), dtype=float),
            g=(np.array(pop.get("G"), dtype=float) if pop.get("G") is not None else None),
        )
        self.snapshots.append(snapshot)
