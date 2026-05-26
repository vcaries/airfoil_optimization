"""Shared pytest fixtures for the aeroforge test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from aeroforge.geometry import Airfoil, NACA4Generator


@pytest.fixture(scope="session")
def naca_0012() -> Airfoil:
    """Return a 120-station-per-surface NACA 0012 airfoil."""
    return NACA4Generator("0012", n_points=120).generate()


@pytest.fixture(scope="session")
def naca_2412() -> Airfoil:
    """Return a 120-station-per-surface NACA 2412 airfoil."""
    return NACA4Generator("2412", n_points=120).generate()


@pytest.fixture
def tmp_dat_path(tmp_path: Path) -> Path:
    """Return a path inside the per-test ``tmp_path`` for ``.dat`` round-trips."""
    return tmp_path / "airfoil.dat"
