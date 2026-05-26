"""Contour smoothing utilities.

Smoothing removes high-frequency noise from imported or algorithmically
generated airfoils (e.g. digitized coordinates) that would otherwise produce
spurious curvature spikes and confuse XFOIL's boundary-layer solver.

Currently a scaffold: the public signatures are stable, the implementations are
planned for a later milestone (see ``ARCHITECTURE.md`` roadmap).
"""

from __future__ import annotations

from aeroforge.core.types import FloatArray


def smooth_savgol(
    x: FloatArray,
    y: FloatArray,
    *,
    window: int = 7,
    polyorder: int = 3,
) -> tuple[FloatArray, FloatArray]:
    """Smooth a contour with a Savitzky-Golay filter.

    Args:
        x: X coordinates in Selig order.
        y: Y coordinates matching ``x``.
        window: Filter window length (odd integer).
        polyorder: Polynomial order of the local fit.

    Returns:
        The smoothed ``(x, y)`` arrays.

    Raises:
        NotImplementedError: Planned for a future milestone.
    """
    raise NotImplementedError("Savitzky-Golay smoothing is planned (see roadmap).")
