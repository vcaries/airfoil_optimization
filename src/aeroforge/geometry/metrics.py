"""Geometric metrics computed from raw airfoil coordinate arrays.

All functions take coordinates in **Selig order** (starting at the trailing
edge, running forward over the upper surface to the leading edge, then back over
the lower surface to the trailing edge) and use only NumPy, so they are fast,
deterministic, and dependency-light. The :class:`Airfoil` class delegates its
geometric properties to these functions.
"""

from __future__ import annotations

import numpy as np

from aeroforge.core.exceptions import InvalidAirfoilError
from aeroforge.core.types import FloatArray
from aeroforge.geometry.operations.discretize import cosine_spacing


def leading_edge_index(x: FloatArray) -> int:
    """Return the index of the leading-edge point (minimum x).

    Args:
        x: X coordinates in Selig order.

    Returns:
        The index of the point with the smallest x coordinate.
    """
    return int(np.argmin(np.asarray(x)))


def _as_monotonic(xs: FloatArray, ys: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Sort a surface by x and drop near-duplicate stations.

    Ensures the arrays are strictly increasing in x so they can be passed to
    :func:`numpy.interp`.

    Args:
        xs: Surface x coordinates.
        ys: Surface y coordinates.

    Returns:
        Monotonic ``(xs, ys)`` arrays.
    """
    order = np.argsort(xs, kind="stable")
    xs, ys = xs[order], ys[order]
    keep = np.concatenate((np.array([True]), np.diff(xs) > 1e-12))
    return xs[keep], ys[keep]


def split_surfaces(
    x: FloatArray, y: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Split Selig coordinates into upper and lower surfaces.

    Both returned surfaces are ordered from the leading edge to the trailing
    edge and are monotonic in x.

    Args:
        x: X coordinates in Selig order.
        y: Y coordinates matching ``x``.

    Returns:
        A tuple ``(x_upper, y_upper, x_lower, y_lower)``.

    Raises:
        InvalidAirfoilError: If fewer than 3 points are supplied.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        raise InvalidAirfoilError("At least 3 points are required to split surfaces.")

    le = leading_edge_index(x)
    # Upper: TE -> LE in Selig order; reverse to get LE -> TE.
    xu, yu = x[le::-1], y[le::-1]
    # Lower: LE -> TE already.
    xl, yl = x[le:], y[le:]
    xu, yu = _as_monotonic(xu, yu)
    xl, yl = _as_monotonic(xl, yl)
    return xu, yu, xl, yl


def _interpolated_surfaces(
    x: FloatArray, y: FloatArray, n: int = 400
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Interpolate both surfaces onto a shared cosine-spaced x grid.

    Args:
        x: X coordinates in Selig order.
        y: Y coordinates matching ``x``.
        n: Number of stations in the shared grid.

    Returns:
        A tuple ``(x_grid, y_upper, y_lower)`` on the common grid.
    """
    xu, yu, xl, yl = split_surfaces(x, y)
    x_lo = max(xu.min(), xl.min())
    x_hi = min(xu.max(), xl.max())
    grid = cosine_spacing(n, x_lo, x_hi)
    yu_i = np.interp(grid, xu, yu)
    yl_i = np.interp(grid, xl, yl)
    return grid, yu_i, yl_i


def max_thickness(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    """Compute the maximum thickness and its chordwise location.

    Args:
        x: X coordinates in Selig order.
        y: Y coordinates matching ``x``.

    Returns:
        A tuple ``(t_max, x_at_t_max)`` where ``t_max`` is the maximum
        thickness as a fraction of chord and ``x_at_t_max`` is its x/c location.
    """
    grid, yu_i, yl_i = _interpolated_surfaces(x, y)
    thickness = yu_i - yl_i
    idx = int(np.argmax(thickness))
    return float(thickness[idx]), float(grid[idx])


def max_camber(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    """Compute the maximum (signed) camber and its chordwise location.

    Camber at a station is the mean of the upper and lower surface ordinates.

    Args:
        x: X coordinates in Selig order.
        y: Y coordinates matching ``x``.

    Returns:
        A tuple ``(c_max, x_at_c_max)`` where ``c_max`` is the camber of largest
        magnitude (sign preserved) and ``x_at_c_max`` is its x/c location.
    """
    grid, yu_i, yl_i = _interpolated_surfaces(x, y)
    camber = 0.5 * (yu_i + yl_i)
    idx = int(np.argmax(np.abs(camber)))
    return float(camber[idx]), float(grid[idx])


def enclosed_area(x: FloatArray, y: FloatArray) -> float:
    """Compute the cross-sectional area enclosed by the contour (shoelace).

    Args:
        x: X coordinates in Selig order.
        y: Y coordinates matching ``x``.

    Returns:
        The enclosed area as a positive float (units of chord squared).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def trailing_edge_gap(x: FloatArray, y: FloatArray) -> float:
    """Compute the trailing-edge gap (distance between first and last points).

    Args:
        x: X coordinates in Selig order.
        y: Y coordinates matching ``x``.

    Returns:
        The Euclidean distance between the first and last coordinate points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.hypot(x[0] - x[-1], y[0] - y[-1]))
