"""Static plot generators for aerodynamic and geometric results.

All functions accept an optional ``ax`` so callers can compose multi-panel
figures, and return the :class:`~matplotlib.axes.Axes` they wrote to. None of
them call :func:`matplotlib.pyplot.show` -- that is left to the caller, which
keeps the functions usable in both interactive and headless contexts.

Requires the ``viz`` extra (``pip install aeroforge[viz]``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from aeroforge.visualization.style import PORTFOLIO_PALETTE

if TYPE_CHECKING:
    from matplotlib.axes import Axes  # type: ignore[import-not-found]

    from aeroforge.geometry.airfoil import Airfoil
    from aeroforge.solver.xfoil.results import CpDistribution, Polar


def _ensure_axes(ax: Any) -> Axes:
    """Return ``ax`` or create a new one on a fresh figure."""
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    if ax is None:
        _fig, ax = plt.subplots()
    return ax  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def plot_geometry(
    airfoil: Airfoil,
    *,
    ax: Any = None,
    show_chord: bool = True,
    show_markers: bool = True,
    label: str | None = None,
    color: str = PORTFOLIO_PALETTE[0],
) -> Axes:
    """Plot the airfoil contour with a 1:1 aspect ratio.

    Args:
        airfoil: The :class:`Airfoil` to draw.
        ax: Optional matplotlib axes to draw into.
        show_chord: Draw the chord line from leading edge to trailing edge.
        show_markers: Mark the leading and trailing edges with dots.
        label: Optional legend label.
        color: Line color (defaults to the first palette entry).

    Returns:
        The :class:`~matplotlib.axes.Axes` the contour was drawn on.
    """
    axes = _ensure_axes(ax)
    axes.plot(airfoil.x, airfoil.y, color=color, lw=1.6, label=label or airfoil.name)
    le = airfoil.leading_edge
    te = airfoil.trailing_edge
    if show_chord:
        axes.plot([le[0], te[0]], [le[1], te[1]], color="0.5", ls=":", lw=0.8)
    if show_markers:
        axes.plot(*le, "o", color="0.2", ms=4)
        axes.plot(*te, "o", color="0.2", ms=4)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x / c")
    axes.set_ylabel("y / c")
    axes.set_title(airfoil.name)
    return axes


# --------------------------------------------------------------------------- #
# Polar
# --------------------------------------------------------------------------- #
def plot_cl_alpha(polar: Polar, *, ax: Any = None) -> Axes:
    """Plot ``C_l`` vs angle of attack.

    Args:
        polar: The :class:`Polar` to draw.
        ax: Optional matplotlib axes.

    Returns:
        The :class:`~matplotlib.axes.Axes` written to.
    """
    axes = _ensure_axes(ax)
    alpha = [p.operating_point.alpha for p in polar.points]
    cl = [p.cl for p in polar.points]
    axes.plot(alpha, cl, "-o", color=PORTFOLIO_PALETTE[0], lw=1.5, ms=4)
    axes.set_xlabel(r"$\alpha$ [deg]")
    axes.set_ylabel(r"$C_l$")
    axes.set_title(r"Lift curve")
    axes.axhline(0.0, color="0.7", lw=0.5)
    return axes


def plot_drag_polar(polar: Polar, *, ax: Any = None) -> Axes:
    """Plot the drag polar ``C_l`` vs ``C_d``.

    Args:
        polar: The :class:`Polar` to draw.
        ax: Optional matplotlib axes.

    Returns:
        The :class:`~matplotlib.axes.Axes` written to.
    """
    axes = _ensure_axes(ax)
    cd = [p.cd for p in polar.points]
    cl = [p.cl for p in polar.points]
    axes.plot(cd, cl, "-o", color=PORTFOLIO_PALETTE[1], lw=1.5, ms=4)
    axes.set_xlabel(r"$C_d$")
    axes.set_ylabel(r"$C_l$")
    axes.set_title(r"Drag polar")
    return axes


def plot_polar(polar: Polar, *, axes: Sequence[Any] | None = None) -> Sequence[Axes]:
    r"""Plot a two-panel ``(C_l/\alpha, C_l/C_d)`` polar.

    Args:
        polar: The :class:`Polar` to draw.
        axes: Optional 2-tuple of pre-built axes for the two panels.

    Returns:
        The two :class:`~matplotlib.axes.Axes` used, in
        ``(cl_alpha_axis, drag_polar_axis)`` order.
    """
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    if axes is None:
        _fig, axes_arr = plt.subplots(1, 2, figsize=(10.5, 4.2))
        axes_seq: Sequence[Axes] = axes_arr
    else:
        axes_seq = axes
    plot_cl_alpha(polar, ax=axes_seq[0])
    plot_drag_polar(polar, ax=axes_seq[1])
    return axes_seq


# --------------------------------------------------------------------------- #
# Cp distribution
# --------------------------------------------------------------------------- #
def plot_cp(cp: CpDistribution, *, ax: Any = None) -> Axes:
    """Plot a chordwise pressure-coefficient distribution with inverted y-axis.

    The y-axis is inverted so that suction peaks point up, which is the
    conventional aerodynamic presentation. Upper and lower surfaces are
    plotted in different colors.

    Args:
        cp: The :class:`CpDistribution` to draw.
        ax: Optional matplotlib axes.

    Returns:
        The :class:`~matplotlib.axes.Axes` written to.
    """
    axes = _ensure_axes(ax)
    x = np.asarray(cp.x, dtype=float)
    cp_arr = np.asarray(cp.cp, dtype=float)
    # Selig-ordered Cp: split at LE (argmin x) into upper (TE -> LE, reverse it)
    # and lower (LE -> TE).
    le = int(np.argmin(x))
    axes.plot(
        x[: le + 1],
        cp_arr[: le + 1],
        color=PORTFOLIO_PALETTE[0],
        lw=1.5,
        label="upper",
    )
    axes.plot(
        x[le:],
        cp_arr[le:],
        color=PORTFOLIO_PALETTE[1],
        lw=1.5,
        label="lower",
    )
    axes.invert_yaxis()
    axes.set_xlabel("x / c")
    axes.set_ylabel(r"$C_p$")
    op = cp.operating_point
    axes.set_title(rf"$C_p$ at $\alpha={op.alpha:.2f}$ deg, $Re={op.reynolds:.1e}$")
    axes.legend(loc="best", fontsize="small")
    return axes


# --------------------------------------------------------------------------- #
# Convergence history
# --------------------------------------------------------------------------- #
def plot_convergence_history(
    history: Sequence[float],
    *,
    ax: Any = None,
    label: str | None = None,
) -> Axes:
    """Plot an optimization or solver convergence history.

    Args:
        history: Sequence of metric values, one per iteration / generation.
        ax: Optional matplotlib axes.
        label: Optional legend label.

    Returns:
        The :class:`~matplotlib.axes.Axes` written to.
    """
    axes = _ensure_axes(ax)
    values = np.asarray(history, dtype=float)
    gens = np.arange(values.size)
    axes.plot(gens, values, "-o", color=PORTFOLIO_PALETTE[0], lw=1.5, ms=4, label=label)
    axes.set_xlabel("generation")
    axes.set_ylabel("objective")
    axes.set_title("Convergence history")
    if label is not None:
        axes.legend(loc="best")
    return axes
