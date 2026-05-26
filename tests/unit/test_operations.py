"""Unit tests for :mod:`aeroforge.geometry.operations`."""

from __future__ import annotations

import numpy as np
import pytest

from aeroforge.core.exceptions import GeometryError
from aeroforge.geometry.operations import (
    cosine_spacing,
    linear_spacing,
    rotate,
    scale,
    translate,
)


# --------------------------------------------------------------------------- #
# Spacing
# --------------------------------------------------------------------------- #
class TestSpacing:
    """Behaviour of the cosine and linear point distributions."""

    @pytest.mark.parametrize("n", [2, 10, 100, 500])
    def test_cosine_spacing_is_monotonic(self, n: int) -> None:
        """Cosine spacing is strictly increasing on the interval."""
        xs = cosine_spacing(n, 0.0, 1.0)
        assert xs.size == n
        assert xs[0] == pytest.approx(0.0)
        assert xs[-1] == pytest.approx(1.0)
        assert np.all(np.diff(xs) > 0)

    def test_cosine_spacing_clusters_at_endpoints(self) -> None:
        """Endpoint spacing is much smaller than mid-interval spacing."""
        xs = cosine_spacing(50, 0.0, 1.0)
        d = np.diff(xs)
        assert d[0] < d[len(d) // 2]
        assert d[-1] < d[len(d) // 2]

    def test_linear_spacing_uniform(self) -> None:
        """Linear spacing is uniform."""
        xs = linear_spacing(11, 0.0, 1.0)
        d = np.diff(xs)
        assert np.allclose(d, d[0])

    @pytest.mark.parametrize("fn", [cosine_spacing, linear_spacing])
    def test_too_few_points_raises(self, fn) -> None:  # noqa: ANN001
        """Asking for fewer than 2 points raises :class:`GeometryError`."""
        with pytest.raises(GeometryError):
            fn(1)


# --------------------------------------------------------------------------- #
# Affine transforms
# --------------------------------------------------------------------------- #
class TestAffineTransforms:
    """Translate/scale/rotate operate correctly on raw arrays."""

    def test_translate_shifts_coordinates(self) -> None:
        """Translation shifts every point by `(dx, dy)`."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 0.0])
        xt, yt = translate(x, y, 3.0, -1.5)
        np.testing.assert_allclose(xt, x + 3.0)
        np.testing.assert_allclose(yt, y - 1.5)

    def test_scale_about_origin(self) -> None:
        """Scaling about the origin multiplies coordinates uniformly."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0])
        xs, ys = scale(x, y, 2.0)
        np.testing.assert_allclose(xs, 2.0 * x)
        np.testing.assert_allclose(ys, 2.0 * y)

    def test_scale_about_arbitrary_origin(self) -> None:
        """Scaling about ``origin`` leaves that point fixed."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        xs, ys = scale(x, y, 3.0, origin=(1.0, 1.0))
        # First point is the fixed point.
        assert xs[0] == pytest.approx(1.0)
        assert ys[0] == pytest.approx(1.0)

    def test_rotate_90_degrees(self) -> None:
        """Rotating ``(1, 0)`` by 90 deg CCW yields ``(0, 1)``."""
        x = np.array([1.0])
        y = np.array([0.0])
        xr, yr = rotate(x, y, 90.0)
        assert xr[0] == pytest.approx(0.0, abs=1e-12)
        assert yr[0] == pytest.approx(1.0, abs=1e-12)

    def test_rotate_full_turn_is_identity(self) -> None:
        """Rotating by 360 deg returns the original coordinates."""
        x = np.array([1.0, 0.5, -0.5])
        y = np.array([0.2, -0.4, 0.1])
        xr, yr = rotate(x, y, 360.0)
        np.testing.assert_allclose(xr, x, atol=1e-12)
        np.testing.assert_allclose(yr, y, atol=1e-12)
