"""Chordwise point distributions and contour repaneling.

Good paneling is critical for XFOIL accuracy: points must cluster near the
leading and trailing edges where curvature and pressure gradients are largest.
Cosine spacing is the standard choice and is used by the built-in generators.
"""

from __future__ import annotations

import numpy as np

from aeroforge.core.exceptions import GeometryError
from aeroforge.core.types import FloatArray


def cosine_spacing(n: int, x_start: float = 0.0, x_end: float = 1.0) -> FloatArray:
    """Generate ``n`` points with cosine (half-cosine) clustering at both ends.

    Points are dense near ``x_start`` and ``x_end`` and sparse in the middle,
    which concentrates resolution at the leading and trailing edges.

    Args:
        n: Number of points to generate. Must be ``>= 2``.
        x_start: Lower bound of the interval.
        x_end: Upper bound of the interval.

    Returns:
        A monotonically increasing array of ``n`` values in ``[x_start, x_end]``.

    Raises:
        GeometryError: If ``n < 2``.
    """
    if n < 2:
        raise GeometryError(f"cosine_spacing needs n >= 2, got {n}.")
    beta = np.linspace(0.0, np.pi, n)
    unit = 0.5 * (1.0 - np.cos(beta))  # 0 -> 1 with end clustering
    result = x_start + (x_end - x_start) * unit
    return np.asarray(result, dtype=np.float64)


def linear_spacing(n: int, x_start: float = 0.0, x_end: float = 1.0) -> FloatArray:
    """Generate ``n`` uniformly spaced points in ``[x_start, x_end]``.

    Args:
        n: Number of points. Must be ``>= 2``.
        x_start: Lower bound of the interval.
        x_end: Upper bound of the interval.

    Returns:
        A monotonically increasing array of ``n`` evenly spaced values.

    Raises:
        GeometryError: If ``n < 2``.
    """
    if n < 2:
        raise GeometryError(f"linear_spacing needs n >= 2, got {n}.")
    return np.linspace(x_start, x_end, n, dtype=np.float64)


def repanel(
    x: FloatArray,
    y: FloatArray,
    n_points: int,
    *,
    smoothing: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Resample a closed airfoil contour onto a cosine arc-length distribution.

    The contour is re-parameterized by normalized arc length and interpolated
    with a periodic-free cubic spline, then sampled with cosine spacing. This
    is the recommended way to bring an imported ``.dat`` file up to the panel
    count XFOIL expects.

    Args:
        x: X coordinates in Selig order (TE -> upper -> LE -> lower -> TE).
        y: Y coordinates matching ``x``.
        n_points: Desired number of output points.
        smoothing: Spline smoothing factor ``s`` (0 interpolates exactly).

    Returns:
        A ``(x_new, y_new)`` tuple of resampled coordinates.

    Note:
        Uses :mod:`scipy.interpolate`, imported lazily so the geometry package
        stays importable in NumPy-only environments.
    """
    # Lazy import keeps SciPy out of the module-import critical path.
    from scipy.interpolate import splev, splprep

    pts = np.vstack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    tck, _ = splprep(pts, s=smoothing, per=False)
    u_new = cosine_spacing(n_points)
    x_new, y_new = splev(u_new, tck)
    return np.asarray(x_new), np.asarray(y_new)
