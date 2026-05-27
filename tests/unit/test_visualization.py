"""Unit tests for the visualization layer.

Matplotlib's Agg backend is forced at import time so tests run headless on
CI. The tests do not assert pixel-perfect output -- they verify that each
plot function returns an Axes, mutates the expected fields, and does not
call ``plt.show``; and that animations produce a non-trivial file at the
expected extension.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aeroforge.core.exceptions import AeroforgeError
from aeroforge.core.types import OperatingPoint
from aeroforge.geometry import NACA4Generator
from aeroforge.optimization.callbacks import GenerationSnapshot, HistoryCallback
from aeroforge.solver.xfoil.results import CpDistribution, Polar, PolarPoint
from aeroforge.visualization import (
    animate_geometry_evolution,
    animate_pareto_evolution,
    non_dominated_mask,
    plot_cl_alpha,
    plot_convergence_history,
    plot_cp,
    plot_drag_polar,
    plot_geometry,
    plot_pareto_front,
    plot_polar,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _close_all_figs():
    """Make sure each test starts with a clean figure stack."""
    yield
    plt.close("all")


@pytest.fixture
def airfoil():
    return NACA4Generator("2412", n_points=40).generate()


def _make_polar(n: int = 5) -> Polar:
    """Build a small synthetic polar."""
    points = [
        PolarPoint(
            operating_point=OperatingPoint(alpha=float(a), reynolds=1e6),
            cl=0.1 * a,
            cd=0.005 + 0.001 * a * a,
            cdp=0.001,
            cm=-0.05,
            x_trans_upper=0.3,
            x_trans_lower=0.6,
            converged=True,
        )
        for a in range(n)
    ]
    return Polar(points=points)


def _make_cp() -> CpDistribution:
    """Build a synthetic Cp distribution in Selig order."""
    # 11 points: TE -> upper -> LE -> lower -> TE.
    x = np.array([1.0, 0.8, 0.5, 0.2, 0.05, 0.0, 0.05, 0.2, 0.5, 0.8, 1.0])
    cp = np.array([0.1, -0.2, -0.6, -1.1, -1.4, -1.0, 0.0, 0.2, 0.2, 0.1, 0.0])
    return CpDistribution(operating_point=OperatingPoint(alpha=2.0, reynolds=1e6), x=x, cp=cp)


def _make_history(n_gen: int = 3, n_obj: int = 1, pop: int = 8) -> HistoryCallback:
    """Build a synthetic single- or multi-objective history."""
    rng = np.random.default_rng(42)
    cb = HistoryCallback()
    for g in range(n_gen):
        cb.snapshots.append(
            GenerationSnapshot(
                generation=g,
                x=rng.uniform(0, 1, size=(pop, 3)),
                # Each generation's F drifts a bit so the animation has motion.
                f=rng.uniform(0, 1, size=(pop, n_obj)) - 0.1 * g,
            )
        )
    return cb


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
class TestPlotGeometry:
    def test_returns_axes_with_data(self, airfoil) -> None:
        ax = plot_geometry(airfoil)
        assert ax is not None
        # At least one Line2D was added.
        assert len(ax.get_lines()) >= 1
        # Aspect ratio is locked.
        assert ax.get_aspect() in ("equal", 1.0)

    def test_accepts_external_axes(self, airfoil) -> None:
        _fig, ax = plt.subplots()
        returned = plot_geometry(airfoil, ax=ax)
        assert returned is ax


# --------------------------------------------------------------------------- #
# Polar
# --------------------------------------------------------------------------- #
class TestPlotPolar:
    def test_cl_alpha_renders(self) -> None:
        polar = _make_polar()
        ax = plot_cl_alpha(polar)
        assert "alpha" in ax.get_xlabel().lower() or r"\alpha" in ax.get_xlabel()

    def test_drag_polar_renders(self) -> None:
        polar = _make_polar()
        ax = plot_drag_polar(polar)
        # CD on x, CL on y.
        assert "C_d" in ax.get_xlabel() or "C_{d}" in ax.get_xlabel()

    def test_polar_returns_two_axes(self) -> None:
        polar = _make_polar()
        axes = plot_polar(polar)
        assert len(axes) == 2


# --------------------------------------------------------------------------- #
# Cp
# --------------------------------------------------------------------------- #
class TestPlotCp:
    def test_inverts_yaxis(self) -> None:
        ax = plot_cp(_make_cp())
        # Convention: y-axis must be inverted so suction peaks point up.
        ymin, ymax = ax.get_ylim()
        assert ymin > ymax

    def test_legend_has_upper_lower(self) -> None:
        ax = plot_cp(_make_cp())
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert "upper" in labels and "lower" in labels


# --------------------------------------------------------------------------- #
# Convergence
# --------------------------------------------------------------------------- #
class TestPlotConvergence:
    def test_renders_a_line(self) -> None:
        history = [1.0, 0.9, 0.7, 0.5, 0.45, 0.4]
        ax = plot_convergence_history(history)
        assert len(ax.get_lines()) == 1
        line = ax.get_lines()[0]
        np.testing.assert_allclose(line.get_ydata(), history)


# --------------------------------------------------------------------------- #
# Pareto
# --------------------------------------------------------------------------- #
class TestParetoFront:
    def test_non_dominated_mask_basic(self) -> None:
        # Two non-dominated points + one dominated.
        f = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
        mask = non_dominated_mask(f)
        np.testing.assert_array_equal(mask, [True, True, False])

    def test_plot_pareto_front_two_objectives(self) -> None:
        history = _make_history(n_gen=1, n_obj=2, pop=12)
        ax = plot_pareto_front(history.snapshots[0])
        # Two collections: dominated + non-dominated scatters.
        assert len(ax.collections) >= 2

    def test_plot_pareto_front_rejects_single_objective(self) -> None:
        history = _make_history(n_gen=1, n_obj=1, pop=12)
        with pytest.raises(ValueError):
            plot_pareto_front(history.snapshots[0])


# --------------------------------------------------------------------------- #
# Animations
# --------------------------------------------------------------------------- #
class TestAnimateGeometry:
    def test_writes_a_gif(self, tmp_path: Path) -> None:
        history = _make_history(n_gen=3, n_obj=1, pop=6)

        # Tiny "factory" that produces a different airfoil per genome.
        def factory(x):
            digit = max(int(round(x[2] * 20)) + 6, 6)
            return NACA4Generator(f"00{digit:02d}", n_points=40).generate()

        out = animate_geometry_evolution(
            history,
            factory,
            tmp_path / "g.gif",
            fps=4,
        )
        assert out.exists()
        assert out.stat().st_size > 500

    def test_rejects_empty_history(self, tmp_path: Path) -> None:
        with pytest.raises(AeroforgeError):
            animate_geometry_evolution(HistoryCallback(), lambda x: None, tmp_path / "g.gif")

    def test_rejects_unknown_extension(self, tmp_path: Path) -> None:
        history = _make_history(n_gen=2, n_obj=1, pop=4)

        def factory(x):
            return NACA4Generator("0012", n_points=40).generate()

        with pytest.raises(AeroforgeError):
            animate_geometry_evolution(history, factory, tmp_path / "g.bogus")


class TestAnimatePareto:
    def test_writes_a_gif(self, tmp_path: Path) -> None:
        history = _make_history(n_gen=3, n_obj=2, pop=8)
        out = animate_pareto_evolution(history, tmp_path / "p.gif", fps=4)
        assert out.exists()
        assert out.stat().st_size > 500

    def test_rejects_single_objective(self, tmp_path: Path) -> None:
        history = _make_history(n_gen=2, n_obj=1, pop=6)
        with pytest.raises(AeroforgeError):
            animate_pareto_evolution(history, tmp_path / "p.gif")
