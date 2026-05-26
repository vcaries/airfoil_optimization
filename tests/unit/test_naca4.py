"""Unit tests for :class:`aeroforge.geometry.NACA4Generator`."""

from __future__ import annotations

import numpy as np
import pytest

from aeroforge.core.exceptions import GeneratorError
from aeroforge.core.types import TrailingEdge
from aeroforge.geometry import Airfoil, NACA4Generator


# --------------------------------------------------------------------------- #
# Designation parsing
# --------------------------------------------------------------------------- #
class TestDesignationParsing:
    """Validate ``_parse`` and constructor input handling."""

    def test_naca0012_decomposition(self) -> None:
        """The 0012 designation yields zero camber and 12% thickness."""
        gen = NACA4Generator("0012")
        # Access internals through generate() side effects + a fresh parser call.
        m, p, t = NACA4Generator._parse("0012")
        assert m == 0.0
        assert p == 0.0
        assert t == pytest.approx(0.12)
        assert gen.name == "NACA 0012"

    def test_naca2412_decomposition(self) -> None:
        """The 2412 designation yields 2% camber at 40% chord and 12% thickness."""
        m, p, t = NACA4Generator._parse("2412")
        assert m == pytest.approx(0.02)
        assert p == pytest.approx(0.4)
        assert t == pytest.approx(0.12)

    @pytest.mark.parametrize("bad", ["", "12", "12345", "abcd", "00 12"])
    def test_rejects_invalid_designation(self, bad: str) -> None:
        """Non-4-digit designations raise :class:`GeneratorError`."""
        with pytest.raises(GeneratorError):
            NACA4Generator(bad)

    def test_rejects_camber_with_zero_position(self) -> None:
        """Nonzero camber with camber position at LE is rejected."""
        with pytest.raises(GeneratorError):
            NACA4Generator("2012")  # m > 0, p = 0

    def test_rejects_too_few_points(self) -> None:
        """``n_points`` must be at least 2."""
        with pytest.raises(GeneratorError):
            NACA4Generator("0012", n_points=1)


# --------------------------------------------------------------------------- #
# Generated geometry
# --------------------------------------------------------------------------- #
class TestGeneratedGeometry:
    """Validate the coordinates produced by ``generate``."""

    def test_returns_airfoil_instance(self, naca_0012: Airfoil) -> None:
        """``generate`` returns an :class:`Airfoil`."""
        assert isinstance(naca_0012, Airfoil)

    def test_point_count_is_2n_minus_1(self) -> None:
        """The Selig contour has ``2 * n_points - 1`` points."""
        n = 80
        airfoil = NACA4Generator("0012", n_points=n).generate()
        assert airfoil.n_points == 2 * n - 1

    def test_starts_at_trailing_edge(self, naca_0012: Airfoil) -> None:
        """The first point is at the upper-surface trailing edge (x ~ 1)."""
        assert naca_0012.x[0] == pytest.approx(1.0, abs=1e-9)
        # And the y at TE is small (zero for closed-TE NACA0012).
        assert abs(naca_0012.y[0]) < 1e-6

    def test_ends_at_trailing_edge(self, naca_0012: Airfoil) -> None:
        """The last point is at the lower-surface trailing edge (x ~ 1)."""
        assert naca_0012.x[-1] == pytest.approx(1.0, abs=1e-9)

    def test_passes_through_leading_edge(self, naca_0012: Airfoil) -> None:
        """The contour reaches ``x = 0`` (the leading edge) exactly once."""
        i = int(np.argmin(naca_0012.x))
        assert naca_0012.x[i] == pytest.approx(0.0, abs=1e-12)
        assert abs(naca_0012.y[i]) < 1e-9

    def test_x_is_finite(self, naca_0012: Airfoil) -> None:
        """Coordinates are finite floats."""
        assert np.all(np.isfinite(naca_0012.x))
        assert np.all(np.isfinite(naca_0012.y))

    def test_no_duplicate_leading_edge(self) -> None:
        """The LE node is shared, not duplicated, in the merged contour."""
        airfoil = NACA4Generator("2412", n_points=50).generate()
        i = int(np.argmin(airfoil.x))
        # Neighbouring x values differ from the LE x by a nonzero amount.
        assert not np.isclose(airfoil.x[i], airfoil.x[i - 1])
        assert not np.isclose(airfoil.x[i], airfoil.x[i + 1])


# --------------------------------------------------------------------------- #
# Symmetric airfoil (NACA 0012)
# --------------------------------------------------------------------------- #
class TestSymmetricAirfoil:
    """NACA 0012 is symmetric about the chord line."""

    def test_max_thickness_is_about_twelve_percent(self, naca_0012: Airfoil) -> None:
        """Maximum thickness is ~12% of chord."""
        assert naca_0012.max_thickness == pytest.approx(0.12, abs=2e-3)

    def test_max_thickness_location_near_30_percent(self, naca_0012: Airfoil) -> None:
        """Maximum thickness occurs near 30% chord (classic NACA result)."""
        assert 0.25 <= naca_0012.max_thickness_location <= 0.35

    def test_camber_is_zero(self, naca_0012: Airfoil) -> None:
        """A symmetric airfoil has zero camber."""
        assert abs(naca_0012.max_camber) < 1e-4

    def test_upper_lower_are_mirror_image(self, naca_0012: Airfoil) -> None:
        """For matching stations, ``y_upper = -y_lower``."""
        xu, yu, xl, yl = naca_0012.surfaces()
        # Interpolate the lower surface onto the upper-surface stations to
        # compare at common x.
        yl_at_xu = np.interp(xu, xl, yl)
        assert np.allclose(yu, -yl_at_xu, atol=1e-6)


# --------------------------------------------------------------------------- #
# Cambered airfoil (NACA 2412)
# --------------------------------------------------------------------------- #
class TestCamberedAirfoil:
    """NACA 2412 should hit its design camber and thickness."""

    def test_max_thickness_is_about_twelve_percent(self, naca_2412: Airfoil) -> None:
        """Maximum thickness is ~12% of chord."""
        assert naca_2412.max_thickness == pytest.approx(0.12, abs=3e-3)

    def test_max_camber_is_about_two_percent(self, naca_2412: Airfoil) -> None:
        """Maximum camber is ~2% of chord and positive (upper-side hump)."""
        assert naca_2412.max_camber == pytest.approx(0.02, abs=2e-3)

    def test_max_camber_location_near_40_percent(self, naca_2412: Airfoil) -> None:
        """Maximum camber occurs near 40% chord."""
        assert 0.35 <= naca_2412.max_camber_location <= 0.45

    def test_upper_above_lower(self, naca_2412: Airfoil) -> None:
        """Upper surface is strictly above the lower surface (positive thickness)."""
        xu, yu, xl, yl = naca_2412.surfaces()
        yl_at_xu = np.interp(xu, xl, yl)
        # Drop endpoints where thickness is zero (LE, TE) before strict-positive check.
        assert np.all((yu - yl_at_xu)[1:-1] > 0.0)


# --------------------------------------------------------------------------- #
# Trailing-edge convention
# --------------------------------------------------------------------------- #
class TestTrailingEdge:
    """Closed-TE airfoils have zero gap; open-TE airfoils have a small gap."""

    def test_closed_te_has_zero_gap(self) -> None:
        """The closed-TE polynomial yields a zero TE gap."""
        airfoil = NACA4Generator("0012", n_points=80, trailing_edge=TrailingEdge.CLOSED).generate()
        assert airfoil.trailing_edge_gap == pytest.approx(0.0, abs=1e-9)

    def test_open_te_has_finite_gap(self) -> None:
        """The classic open-TE polynomial yields a small but nonzero TE gap."""
        airfoil = NACA4Generator("0012", n_points=80, trailing_edge=TrailingEdge.OPEN).generate()
        assert airfoil.trailing_edge_gap > 1e-4
        assert airfoil.trailing_edge_gap < 5e-3  # bounded by classic NACA formula


# --------------------------------------------------------------------------- #
# Stability across resolutions
# --------------------------------------------------------------------------- #
class TestResolutionStability:
    """Refining ``n_points`` does not change the geometry materially."""

    def test_max_thickness_is_resolution_invariant(self) -> None:
        """Same designation -> same max thickness at coarse and fine resolution."""
        coarse = NACA4Generator("2412", n_points=40).generate().max_thickness
        fine = NACA4Generator("2412", n_points=200).generate().max_thickness
        assert abs(coarse - fine) < 5e-3
