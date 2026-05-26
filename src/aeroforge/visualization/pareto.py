"""Pareto-front plotting helpers.

Requires the ``viz`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aeroforge.optimization.callbacks import GenerationSnapshot


def plot_pareto_front(snapshot: GenerationSnapshot, *, ax: Any = None) -> Any:
    """Plot the non-dominated front from one generation snapshot.

    Args:
        snapshot: A :class:`GenerationSnapshot` captured by
            :class:`HistoryCallback`.
        ax: Optional matplotlib axes to draw into.

    Returns:
        The :class:`~matplotlib.axes.Axes` written to.

    Raises:
        NotImplementedError: Implementation planned for milestone M5.
    """
    raise NotImplementedError("plot_pareto_front (planned, M5).")
