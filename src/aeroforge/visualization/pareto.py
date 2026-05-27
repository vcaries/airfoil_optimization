"""Pareto-front plotting helpers.

Requires the ``viz`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from aeroforge.core.types import FloatArray
from aeroforge.visualization.style import PORTFOLIO_PALETTE

if TYPE_CHECKING:
    from matplotlib.axes import Axes  # type: ignore[import-not-found]

    from aeroforge.optimization.callbacks import GenerationSnapshot


def non_dominated_mask(f: FloatArray) -> npt.NDArray[np.bool_]:
    """Return a boolean mask of non-dominated rows in an objective matrix.

    A row ``a`` is dominated by another row ``b`` iff ``b`` is no worse than
    ``a`` on every objective and strictly better on at least one. pymoo
    minimises by convention, so smaller is better.

    Args:
        f: ``(pop, n_obj)`` objective matrix.

    Returns:
        A boolean array of length ``pop``; ``True`` means non-dominated.
    """
    f = np.asarray(f, dtype=float)
    n = f.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # Vectorised "any other row dominates i?"
        dominates = np.all(f <= f[i], axis=1) & np.any(f < f[i], axis=1)
        if np.any(dominates):
            keep[i] = False
    return keep


def plot_pareto_front(
    snapshot: GenerationSnapshot,
    *,
    ax: Any = None,
    xlabel: str = r"$f_1$",
    ylabel: str = r"$f_2$",
) -> Axes:
    """Plot the population scatter with the non-dominated front highlighted.

    Args:
        snapshot: A :class:`GenerationSnapshot` captured by
            :class:`HistoryCallback`.
        ax: Optional matplotlib axes.
        xlabel: Label for the first objective.
        ylabel: Label for the second objective.

    Returns:
        The :class:`~matplotlib.axes.Axes` written to.

    Raises:
        ValueError: If the snapshot has fewer than two objectives.
    """
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    if ax is None:
        _fig, ax = plt.subplots()
    assert ax is not None
    f = np.asarray(snapshot.f, dtype=float)
    if f.ndim != 2 or f.shape[1] < 2:
        raise ValueError(
            f"plot_pareto_front needs a snapshot with >= 2 objectives, got shape={f.shape}."
        )

    mask = non_dominated_mask(f)
    ax.scatter(
        f[~mask, 0],
        f[~mask, 1],
        s=18,
        color="0.65",
        alpha=0.7,
        label="dominated",
    )
    ax.scatter(
        f[mask, 0],
        f[mask, 1],
        s=42,
        color=PORTFOLIO_PALETTE[3],
        edgecolors="white",
        lw=0.8,
        label="non-dominated",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Pareto front (gen {snapshot.generation})")
    ax.legend(loc="best", fontsize="small")
    return ax  # type: ignore[no-any-return]
