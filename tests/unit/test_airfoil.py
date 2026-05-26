"""Unit tests for :class:`aeroforge.geometry.Airfoil`."""

from __future__ import annotations

import numpy as np
import pytest

from aeroforge.core.exceptions import InvalidAirfoilError
from aeroforge.geometry import Airfoil


class TestConstruction:
    """Constructor validation."""

    def test_rejects_mismatched_shapes(self) -> None:
        """Mismatched x/y lengths are rejected."""
        with pytest.raises(InvalidAirfoilError):
            Airfoil(np.zeros(5), np.zeros(6))

    def test_rejects_too_few_points(self) -> None:
        """Fewer than 4 points is rejected."""
        with pytest.raises(InvalidAirfoilError):
            Airfoil(np.array([0.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]))

    def test_rejects_nan_values(self) -> None:
        """NaN values in coordinates are rejected."""
        with pytest.raises(InvalidAirfoilError):
            Airfoil(
                np.array([1.0, 0.5, np.nan, 0.5, 1.0]),
                np.array([0.0, 0.05, 0.0, -0.05, 0.0]),
            )


class TestProperties:
    """Coordinate access and derived metric properties."""

    def test_n_points_matches_array_size(self, naca_0012):
        assert naca_0012.n_points == naca_0012.x.size
        assert len(naca_0012) == naca_0012.n_points

    def test_coordinates_stack(self, naca_0012):
        coords = naca_0012.coordinates
        assert coords.shape == (naca_0012.n_points, 2)
        np.testing.assert_allclose(coords[:, 0], naca_0012.x)
        np.testing.assert_allclose(coords[:, 1], naca_0012.y)

    def test_chord_is_about_unit_for_generated_airfoil(self, naca_2412):
        """For a cambered airfoil min(x) is a few 1e-5 negative; 1e-3 covers it."""
        assert naca_2412.chord == pytest.approx(1.0, abs=1e-3)

    def test_leading_edge_near_origin(self, naca_2412):
        le_x, le_y = naca_2412.leading_edge
        assert abs(le_x) < 5e-3
        assert abs(le_y) < 5e-3


class TestTransforms:
    """Affine transforms return new instances and compose correctly."""

    def test_translate_round_trip(self, naca_0012):
        moved = naca_0012.translated(2.0, -3.5).translated(-2.0, 3.5)
        np.testing.assert_allclose(moved.x, naca_0012.x, atol=1e-12)
        np.testing.assert_allclose(moved.y, naca_0012.y, atol=1e-12)

    def test_scale_doubles_thickness(self, naca_0012):
        bigger = naca_0012.scaled(2.0)
        assert bigger.max_thickness == pytest.approx(2.0 * naca_0012.max_thickness, abs=1e-3)

    def test_rotate_preserves_chord_length(self, naca_2412):
        rotated = naca_2412.rotated(15.0)
        assert rotated.chord == pytest.approx(naca_2412.chord, abs=1e-3)

    def test_returns_new_instance(self, naca_0012):
        original_first = naca_0012.x[0]
        _ = naca_0012.translated(1.0, 0.0)
        assert naca_0012.x[0] == original_first


class TestNormalization:
    """normalized places the chord on the x-axis from (0, 0)."""

    def test_normalized_chord_is_about_unit(self, naca_2412):
        moved = naca_2412.translated(3.0, -1.0).rotated(10.0).scaled(2.5)
        result = moved.normalized()
        assert result.chord == pytest.approx(1.0, abs=2e-3)

    def test_normalized_leading_edge_near_origin(self, naca_2412):
        moved = naca_2412.rotated(20.0).translated(5.0, 0.7)
        result = moved.normalized()
        le_x, le_y = result.leading_edge
        assert abs(le_x) < 1e-2
        assert abs(le_y) < 1e-2


class TestIO:
    """dat round-trip preserves geometry."""

    def test_to_from_dat_round_trip(self, naca_2412, tmp_dat_path):
        naca_2412.to_dat(tmp_dat_path)
        loaded = Airfoil.from_dat(tmp_dat_path)
        np.testing.assert_allclose(loaded.x, naca_2412.x, atol=1e-5)
        np.testing.assert_allclose(loaded.y, naca_2412.y, atol=1e-5)

    def test_from_dat_skips_header(self, tmp_dat_path):
        tmp_dat_path.write_text(
            "My airfoil\n1.0 0.0\n0.5 0.05\n0.0 0.0\n0.5 -0.05\n1.0 0.0\n",
            encoding="utf-8",
        )
        airfoil = Airfoil.from_dat(tmp_dat_path)
        assert airfoil.n_points == 5
        assert "airfoil" in airfoil.name.lower()

    def test_rejects_lednicer_format(self, tmp_dat_path):
        tmp_dat_path.write_text(
            "Lednicer\n61.  61.\n1.0 0.0\n0.5 0.05\n0.0 0.0\n",
            encoding="utf-8",
        )
        with pytest.raises(InvalidAirfoilError):
            Airfoil.from_dat(tmp_dat_path)


class TestDunder:
    """Equality, length, and repr behave as expected."""

    def test_equality_against_copy(self, naca_0012):
        clone = naca_0012.copy()
        assert clone == naca_0012

    def test_equality_against_translated_is_false(self, naca_0012):
        assert naca_0012.translated(0.1, 0.0) != naca_0012

    def test_repr_contains_name(self, naca_2412):
        text = repr(naca_2412)
        assert "NACA 2412" in text
        assert "n_points" in text
