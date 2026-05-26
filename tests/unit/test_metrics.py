"""Unit tests for :mod:`aeroforge.geometry.metrics`."""

from __future__ import annotations

import numpy as np
import pytest

from aeroforge.core.exceptions import InvalidAirfoilError
from aeroforge.geometry import NACA4Generator
from aeroforge.geometry.metrics import (
    enclosed_area,
    leading_edge_index,
    max_camber,
    max_thickness,
    split_surfaces,
)


def test_leading_edge_index_finds_min_x() -> None:
    """``leading_edge_index`` returns the index of the smallest x."""
    x = np.array([1.0, 0.5, 0.0, 0.5, 1.0])
    assert leading_edge_index(x) == 2


def test_split_surfaces_starts_at_leading_edge() -> None:
    """Both surfaces start at the LE and end at the TE."""
    airfoil = NACA4Generator("0012", n_points=40).generate()
    xu, yu, xl, yl = split_surfaces(airfoil.x, airfoil.y)
    assert xu[0] == pytest.approx(0.0, abs=1e-9)
    assert xl[0] == pytest.approx(0.0, abs=1e-9)
    assert xu[-1] == pytest.approx(1.0, abs=1e-9)
    assert xl[-1] == pytest.approx(1.0, abs=1e-9)


def test_split_surfaces_monotonic_in_x() -> None:
    """After splitting, x is strictly monotonic on each surface."""
    airfoil = NACA4Generator("2412", n_points=60).generate()
    xu, _, xl, _ = split_surfaces(airfoil.x, airfoil.y)
    assert np.all(np.diff(xu) > 0)
    assert np.all(np.diff(xl) > 0)


def test_max_thickness_naca0012() -> None:
    """NACA 0012 has ~12% max thickness near 30% chord."""
    airfoil = NACA4Generator("0012", n_points=120).generate()
    t, x_at = max_thickness(airfoil.x, airfoil.y)
    assert t == pytest.approx(0.12, abs=2e-3)
    assert 0.25 <= x_at <= 0.35


def test_max_camber_zero_for_symmetric() -> None:
    """A symmetric airfoil has zero camber."""
    airfoil = NACA4Generator("0012", n_points=80).generate()
    c, _ = max_camber(airfoil.x, airfoil.y)
    assert abs(c) < 1e-4


def test_max_camber_positive_for_cambered() -> None:
    """A 2-series airfoil has positive maximum camber."""
    airfoil = NACA4Generator("2412", n_points=80).generate()
    c, _ = max_camber(airfoil.x, airfoil.y)
    assert c > 0
    assert c == pytest.approx(0.02, abs=2e-3)


def test_enclosed_area_positive() -> None:
    """The enclosed area is positive for a valid contour."""
    airfoil = NACA4Generator("2412", n_points=80).generate()
    area = enclosed_area(airfoil.x, airfoil.y)
    assert area > 0
    # A NACA 4-digit 12%-thickness airfoil encloses roughly 0.08 chord^2.
    assert 0.05 < area < 0.12


def test_split_surfaces_rejects_short_arrays() -> None:
    """Two-point inputs are rejected with a clear error."""
    with pytest.raises(InvalidAirfoilError):
        split_surfaces(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
