"""Unit tests for :class:`aeroforge.solver.xfoil.parser.XfoilOutputParser`."""

from __future__ import annotations

from pathlib import Path

import pytest

from aeroforge.core.exceptions import ParsingError
from aeroforge.core.types import OperatingPoint
from aeroforge.solver.xfoil.parser import XfoilOutputParser

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data"


# --------------------------------------------------------------------------- #
# Polar parsing
# --------------------------------------------------------------------------- #
class TestParsePolar:
    """Validate :meth:`XfoilOutputParser.parse_polar`."""

    def test_parses_all_converged_rows(self):
        polar = XfoilOutputParser.parse_polar(FIXTURE_DIR / "polars" / "naca0012_re1e6.pol")
        # Fixture has 9 rows: alpha = -2, -1, 0, 1, 2, 3, 4, 5, 6.
        assert len(polar) == 9
        assert polar.points[0].operating_point.alpha == pytest.approx(-2.0)
        assert polar.points[-1].operating_point.alpha == pytest.approx(6.0)

    def test_extracts_aerodynamic_coefficients(self):
        polar = XfoilOutputParser.parse_polar(FIXTURE_DIR / "polars" / "naca0012_re1e6.pol")
        zero_alpha = next(p for p in polar.points if p.operating_point.alpha == pytest.approx(0.0))
        # Symmetric NACA 0012 at alpha=0: CL=0, CM=0, finite CD.
        assert zero_alpha.cl == pytest.approx(0.0, abs=1e-6)
        assert zero_alpha.cm == pytest.approx(0.0, abs=1e-6)
        assert 0.0 < zero_alpha.cd < 0.01

    def test_extracts_header_reynolds_mach_ncrit(self):
        polar = XfoilOutputParser.parse_polar(FIXTURE_DIR / "polars" / "naca0012_re1e6.pol")
        op = polar.points[0].operating_point
        assert op.reynolds == pytest.approx(1.0e6, rel=1e-6)
        assert op.mach == pytest.approx(0.0)
        assert op.n_crit == pytest.approx(9.0)

    def test_extracts_transition_locations(self):
        polar = XfoilOutputParser.parse_polar(FIXTURE_DIR / "polars" / "naca0012_re1e6.pol")
        # At alpha=2 deg the fixture has Top_Xtr=0.5314, Bot_Xtr=1.0000.
        alpha2 = next(p for p in polar.points if p.operating_point.alpha == pytest.approx(2.0))
        assert alpha2.x_trans_upper == pytest.approx(0.5314)
        assert alpha2.x_trans_lower == pytest.approx(1.0000)

    def test_empty_polar_is_not_an_error(self):
        polar = XfoilOutputParser.parse_polar(FIXTURE_DIR / "polars" / "empty.pol")
        assert len(polar) == 0
        assert polar.failed == []

    def test_malformed_polar_raises(self):
        with pytest.raises(ParsingError):
            XfoilOutputParser.parse_polar(FIXTURE_DIR / "polars" / "malformed.pol")

    def test_short_row_raises(self, tmp_path: Path):
        bad = tmp_path / "short.pol"
        bad.write_text(
            "header\n"
            "  alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr\n"
            "  ------ -------- --------- --------- -------- -------- --------\n"
            "   0.000   0.0000   0.00543\n",
            encoding="utf-8",
        )
        with pytest.raises(ParsingError):
            XfoilOutputParser.parse_polar(bad)

    def test_lift_to_drag_ratio(self):
        polar = XfoilOutputParser.parse_polar(FIXTURE_DIR / "polars" / "naca0012_re1e6.pol")
        # L/D at alpha=4 deg, NACA 0012, Re=1e6: ~0.4869 / 0.00650 ~= 75.
        alpha4 = next(p for p in polar.points if p.operating_point.alpha == pytest.approx(4.0))
        assert 60.0 < alpha4.lift_to_drag < 90.0


# --------------------------------------------------------------------------- #
# Cp parsing
# --------------------------------------------------------------------------- #
class TestParseCp:
    """Validate :meth:`XfoilOutputParser.parse_cp`."""

    def test_parses_x_and_cp_arrays(self):
        op = OperatingPoint(alpha=2.0, reynolds=1.0e6)
        cp = XfoilOutputParser.parse_cp(FIXTURE_DIR / "cp" / "naca0012_alpha2.cp", op)
        # Fixture has 21 stations.
        assert cp.x.size == 21
        assert cp.cp.size == 21
        # First/last stations are at the trailing edge.
        assert cp.x[0] == pytest.approx(1.0)
        assert cp.x[-1] == pytest.approx(1.0)
        # Mid-range station reaches the LE.
        assert min(cp.x) == pytest.approx(0.0)

    def test_carries_operating_point(self):
        op = OperatingPoint(alpha=2.0, reynolds=1.0e6, mach=0.05)
        cp = XfoilOutputParser.parse_cp(FIXTURE_DIR / "cp" / "naca0012_alpha2.cp", op)
        assert cp.operating_point is op

    def test_empty_cp_raises(self, tmp_path: Path):
        empty = tmp_path / "empty.cp"
        empty.write_text("# x   Cp\n", encoding="utf-8")
        with pytest.raises(ParsingError):
            XfoilOutputParser.parse_cp(empty, OperatingPoint(alpha=0.0))

    def test_malformed_cp_row_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.cp"
        bad.write_text("# x   Cp\n0.5  notanumber\n", encoding="utf-8")
        with pytest.raises(ParsingError):
            XfoilOutputParser.parse_cp(bad, OperatingPoint(alpha=0.0))
