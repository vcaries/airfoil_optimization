"""Static plot generators for aerodynamic and geometric results.

All functions accept an optional ``ax`` so callers can compose multi-panel
figures, and return the :class:`~matplotlib.axes.Axes` they wrote to. None of
them call :func:`matplotlib.pyplot.show` -- that is left to the caller, which
keeps the functions usable in both interactive and headless contexts.

Requires the ``viz`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aeroforge.geometry.airfoil import Airfoil
    from aeroforge.solver.xfoil.results import CpDistribution, Polar


def plot_geometry(airfoil: Airfoil, *, ax: Any = None) -> Any:
    """Plot the airfoil contour with a 1:1 aspect ratio.

    Args:
        airfoil: The :class:`Airfoil` to draw.
        ax: Optional matplotlib axes to draw into.

    Returns:
        The :class:`~matplotlib.axes.Axes` the contour was drawn on.

    Raises:
        NotImplementedError: Implementation planned for milestone M5.
    """
    raise NotImplementedError("plot_geometry (planned, M5).")


def plot_polar(polar: Polar, *, ax: Any = None) -> Any:
    """Plot a ``C_l``/``alpha`` and ``C_l``/``C_d`` polar.

    Args:
        polar: The :class:`Polar` to draw.
        ax: Optional matplotlib axes to draw into.

    Returns:
        The :class:`~matplotlib.axes.Axes` written to.

    Raises:
        NotImplementedError: Implementation planned for milestone M5.
    """
    raise NotImplementedError("plot_polar (planned, M5).")


def plot_cp(cp: CpDistribution, *, ax: Any = None) -> Any:
    """Plot a chordwise pressure-coefficient distribution (inverted y-axis).

    Args:
        cp: The :class:`CpDistribution` to draw.
        ax: Optional matplotlib axes to draw into.

    Returns:
        The :class:`~matplotlib.axes.Axes` written to.

    Raises:
        NotImplementedError: Implementation planned for milestone M5.
    """
    raise NotImplementedError("plot_cp (planned, M5).")


def plot_convergence_history(history: list[float], *, ax: Any = None) -> Any:
    """Plot an optimization or solver convergence history.

    Args:
        history: Sequence of metric values, one per iteration / generation.
        ax: Optional matplotlib axes to draw into.

    Returns:
        The :class:`~matplotlib.axes.Axes` written to.

    Raises:
        NotImplementedError: Implementation planned for milestone M5.
    """
    raise NotImplementedError("plot_convergence_history (planned, M5).")
