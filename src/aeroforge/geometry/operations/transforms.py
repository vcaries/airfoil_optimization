"""Affine geometric transformations on airfoil coordinate arrays.

Functions operate on raw ``(x, y)`` arrays and return new arrays (pure, no
mutation) so they compose cleanly and are trivial to unit test. The
:class:`~aeroforge.geometry.airfoil.Airfoil` class exposes thin wrappers around
them.
"""

from __future__ import annotations

import numpy as np

from aeroforge.core.types import FloatArray


def translate(x: FloatArray, y: FloatArray, dx: float, dy: float) -> tuple[FloatArray, FloatArray]:
    """Translate coordinates by ``(dx, dy)``.

    Args:
        x: X coordinates.
        y: Y coordinates.
        dx: Shift applied to X.
        dy: Shift applied to Y.

    Returns:
        The translated ``(x, y)`` arrays.
    """
    return np.asarray(x) + dx, np.asarray(y) + dy


def scale(
    x: FloatArray,
    y: FloatArray,
    factor: float,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
) -> tuple[FloatArray, FloatArray]:
    """Uniformly scale coordinates about ``origin``.

    Args:
        x: X coordinates.
        y: Y coordinates.
        factor: Scale factor (``1.0`` leaves geometry unchanged).
        origin: Fixed point of the scaling.

    Returns:
        The scaled ``(x, y)`` arrays.
    """
    ox, oy = origin
    xs = (np.asarray(x) - ox) * factor + ox
    ys = (np.asarray(y) - oy) * factor + oy
    return xs, ys


def rotate(
    x: FloatArray,
    y: FloatArray,
    angle_deg: float,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
) -> tuple[FloatArray, FloatArray]:
    """Rotate coordinates by ``angle_deg`` (counter-clockwise) about ``origin``.

    A positive angle corresponds to a nose-up rotation of the airfoil, i.e. an
    increase in geometric angle of attack of ``angle_deg``.

    Args:
        x: X coordinates.
        y: Y coordinates.
        angle_deg: Rotation angle in degrees (counter-clockwise positive).
        origin: Center of rotation.

    Returns:
        The rotated ``(x, y)`` arrays.
    """
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    ox, oy = origin
    xr = np.asarray(x) - ox
    yr = np.asarray(y) - oy
    x_new = cos_t * xr - sin_t * yr + ox
    y_new = sin_t * xr + cos_t * yr + oy
    return x_new, y_new
