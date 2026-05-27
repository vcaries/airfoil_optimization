"""Unit tests for the optimization layer (evaluator + problem + study).

A :class:`FakeXfoil` solver returns synthetic, deterministic results so the
whole pipeline can be exercised without the XFOIL binary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from aeroforge.core.exceptions import ConvergenceError, OptimizationError
from aeroforge.core.types import OperatingPoint
from aeroforge.geometry import NACA4Generator
from aeroforge.geometry.airfoil import Airfoil
from aeroforge.optimization import (
    AirfoilEvaluator,
    AirfoilProblem,
    DesignSpace,
    DesignVariable,
    HistoryCallback,
    MaximizeLiftToDrag,
    MinimizeDrag,
    MinThicknessConstraint,
    OptimizationStudy,
)
from aeroforge.optimization.algorithms import ga
from aeroforge.solver.base import AbstractSolver
from aeroforge.solver.xfoil.results import PolarPoint


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
@dataclass
class FakeXfoil(AbstractSolver):
    """Synthetic solver. Returns a converged result unless ``fail`` is True."""

    fail: bool = False
    cl: float = 0.6
    cd: float = 0.008

    def analyze(self, airfoil: Airfoil, point: OperatingPoint) -> PolarPoint:
        if self.fail:
            raise ConvergenceError("synthetic failure", alpha=point.alpha)
        return PolarPoint(
            operating_point=point,
            cl=self.cl,
            cd=self.cd,
            cdp=self.cd / 8,
            cm=-0.05,
            x_trans_upper=0.3,
            x_trans_lower=0.6,
            converged=True,
        )


def build_naca4(params: dict[str, float]) -> Airfoil:
    """Translate a {m, p, t} design vector to a NACA 4-digit airfoil."""
    m = int(round(params["m"] * 9))
    p = max(int(round(params["p"] * 9)), 1)
    t = max(int(round(params["t"] * 24)) + 6, 6)
    return NACA4Generator(f"{m}{p}{t:02d}", n_points=40).generate()


@pytest.fixture
def design_space() -> DesignSpace:
    return DesignSpace(
        [
            DesignVariable("m", 0.0, 0.9),
            DesignVariable("p", 0.1, 1.0),
            DesignVariable("t", 0.0, 1.0),
        ]
    )


# --------------------------------------------------------------------------- #
# AirfoilEvaluator
# --------------------------------------------------------------------------- #
class TestAirfoilEvaluator:
    def test_full_success_returns_objective_and_constraints(
        self, design_space: DesignSpace
    ) -> None:
        evaluator = AirfoilEvaluator(
            design_space=design_space,
            airfoil_factory=build_naca4,
            solver=FakeXfoil(cl=0.6, cd=0.008),
            operating_point=OperatingPoint(alpha=4.0, reynolds=5e5),
            objectives=[MaximizeLiftToDrag()],
            geometric_constraints=[MinThicknessConstraint(t_min=0.08)],
        )
        f, g = evaluator.evaluate(np.array([0.2, 0.4, 0.5]))
        # L/D = 0.6 / 0.008 = 75; maximize -> minimize -75.
        assert f[0] == pytest.approx(-75.0)
        # Thickness > 8% -> constraint <= 0 (feasible).
        assert g[0] <= 0.0

    def test_geometric_violation_skips_solver(self, design_space: DesignSpace) -> None:
        """When a geometric constraint is violated, the solver is not invoked."""
        solver = FakeXfoil(fail=True)  # would raise if called
        evaluator = AirfoilEvaluator(
            design_space=design_space,
            airfoil_factory=build_naca4,
            # Require an unrealistic minimum thickness so every candidate fails.
            geometric_constraints=[MinThicknessConstraint(t_min=0.99)],
            solver=solver,
            operating_point=OperatingPoint(alpha=4.0, reynolds=5e5),
            objectives=[MaximizeLiftToDrag()],
        )
        f, g = evaluator.evaluate(np.array([0.2, 0.4, 0.5]))
        # Failed candidate: sentinel objective, positive constraint.
        assert f[0] >= 1.0e6
        assert g[0] > 0.0  # constraint violation

    def test_solver_failure_is_swallowed(self, design_space: DesignSpace) -> None:
        """ConvergenceError must not crash the optimization loop."""
        evaluator = AirfoilEvaluator(
            design_space=design_space,
            airfoil_factory=build_naca4,
            solver=FakeXfoil(fail=True),
            operating_point=OperatingPoint(alpha=4.0, reynolds=5e5),
            objectives=[MaximizeLiftToDrag()],
        )
        f, g = evaluator.evaluate(np.array([0.2, 0.4, 0.5]))
        assert f[0] >= 1.0e6
        assert g == []  # no constraints declared

    def test_bad_genome_is_swallowed(self, design_space: DesignSpace) -> None:
        """A genome that the factory cannot build must not crash the loop."""

        def bad_factory(_params: dict[str, float]) -> Airfoil:
            raise ValueError("factory cannot build this")

        evaluator = AirfoilEvaluator(
            design_space=design_space,
            airfoil_factory=bad_factory,
            solver=FakeXfoil(),
            operating_point=OperatingPoint(alpha=4.0, reynolds=5e5),
            objectives=[MaximizeLiftToDrag()],
        )
        f, _ = evaluator.evaluate(np.array([0.5, 0.5, 0.5]))
        assert f[0] >= 1.0e6

    def test_count_helpers(self, design_space: DesignSpace) -> None:
        evaluator = AirfoilEvaluator(
            design_space=design_space,
            airfoil_factory=build_naca4,
            solver=FakeXfoil(),
            operating_point=OperatingPoint(alpha=4.0),
            objectives=[MinimizeDrag(), MaximizeLiftToDrag()],
            geometric_constraints=[MinThicknessConstraint(t_min=0.08)],
        )
        assert evaluator.n_obj == 2
        assert evaluator.n_geometric_constraints == 1
        assert evaluator.n_physical_constraints == 0
        assert evaluator.n_constr == 1


# --------------------------------------------------------------------------- #
# AirfoilProblem
# --------------------------------------------------------------------------- #
class TestAirfoilProblem:
    def test_evaluates_population(self, design_space: DesignSpace) -> None:
        evaluator = AirfoilEvaluator(
            design_space=design_space,
            airfoil_factory=build_naca4,
            solver=FakeXfoil(cl=0.6, cd=0.008),
            operating_point=OperatingPoint(alpha=4.0, reynolds=5e5),
            objectives=[MaximizeLiftToDrag()],
            geometric_constraints=[MinThicknessConstraint(t_min=0.08)],
        )
        problem = AirfoilProblem(evaluator)
        # Build a tiny population.
        pop = np.array(
            [
                [0.2, 0.4, 0.5],
                [0.3, 0.5, 0.4],
                [0.1, 0.6, 0.6],
            ]
        )
        out: dict[str, Any] = {}
        problem._evaluate(pop, out)
        assert out["F"].shape == (3, 1)
        assert out["G"].shape == (3, 1)
        # Solver returns a constant, so every objective is identical.
        assert np.allclose(out["F"], -75.0)

    def test_constraint_free_problem_does_not_emit_g(self, design_space: DesignSpace) -> None:
        evaluator = AirfoilEvaluator(
            design_space=design_space,
            airfoil_factory=build_naca4,
            solver=FakeXfoil(),
            operating_point=OperatingPoint(alpha=4.0),
            objectives=[MinimizeDrag()],
        )
        problem = AirfoilProblem(evaluator)
        out: dict[str, Any] = {}
        problem._evaluate(np.array([[0.2, 0.4, 0.5]]), out)
        assert "F" in out
        assert "G" not in out


# --------------------------------------------------------------------------- #
# OptimizationStudy (end-to-end)
# --------------------------------------------------------------------------- #
class TestOptimizationStudy:
    @pytest.fixture
    def evaluator(self, design_space: DesignSpace) -> AirfoilEvaluator:
        """An evaluator with a non-trivial synthetic objective.

        The synthetic L/D peaks at thickness ~10% and camber ~3%, so the GA
        has something to actually search for.
        """

        @dataclass
        class PeakedXfoil(AbstractSolver):
            def analyze(self, airfoil: Airfoil, point: OperatingPoint) -> PolarPoint:
                t = airfoil.max_thickness
                c = airfoil.max_camber
                ld = 80.0 - 100.0 * (t - 0.10) ** 2 - 200.0 * (c - 0.03) ** 2
                cl = 0.5 + c * 5.0
                cd = cl / max(ld, 1.0)
                return PolarPoint(
                    operating_point=point,
                    cl=cl,
                    cd=cd,
                    cdp=0.001,
                    cm=-0.05,
                    x_trans_upper=0.3,
                    x_trans_lower=0.6,
                    converged=True,
                )

        return AirfoilEvaluator(
            design_space=design_space,
            airfoil_factory=build_naca4,
            solver=PeakedXfoil(),
            operating_point=OperatingPoint(alpha=4.0, reynolds=5e5),
            objectives=[MaximizeLiftToDrag()],
        )

    def test_run_drives_pymoo_to_convergence(self, evaluator: AirfoilEvaluator) -> None:
        study = OptimizationStudy(
            evaluator=evaluator,
            algorithm=ga(pop_size=12),
            n_gen=5,
            seed=42,
        )
        result = study.run()
        # Best L/D should be close to the synthetic ceiling of 80.
        # F = -L/D (because pymoo minimizes), so F should be near -80.
        assert float(result.F[0]) < -70.0
        # All 5 generations captured.
        assert len(study.history) == 5
        # Best-per-gen trace is non-increasing (best stays the best or improves).
        trace = study.history.best_per_generation()
        assert all(trace[i + 1] <= trace[i] + 1e-9 for i in range(len(trace) - 1))

    def test_run_is_reproducible_under_same_seed(self, evaluator: AirfoilEvaluator) -> None:
        result_a = OptimizationStudy(
            evaluator=evaluator, algorithm=ga(pop_size=10), n_gen=3, seed=7
        ).run()
        result_b = OptimizationStudy(
            evaluator=evaluator, algorithm=ga(pop_size=10), n_gen=3, seed=7
        ).run()
        np.testing.assert_allclose(result_a.X, result_b.X)
        np.testing.assert_allclose(result_a.F, result_b.F)

    def test_checkpoint_round_trip(self, evaluator: AirfoilEvaluator, tmp_path: Path) -> None:
        ckpt = tmp_path / "study.pkl"
        study = OptimizationStudy(
            evaluator=evaluator,
            algorithm=ga(pop_size=8),
            n_gen=3,
            seed=11,
            checkpoint_path=ckpt,
        )
        study.run()
        assert ckpt.exists()

        # New study reloads the checkpoint.
        fresh = OptimizationStudy(
            evaluator=evaluator,
            algorithm=ga(pop_size=8),
            n_gen=3,
            seed=11,
            checkpoint_path=ckpt,
        )
        fresh.load_checkpoint(ckpt)
        assert len(fresh.history) == 3
        # Snapshot shapes survive the round trip.
        assert fresh.history.snapshots[0].x.shape == (8, 3)

    def test_resume_requires_checkpoint_path(self, evaluator: AirfoilEvaluator) -> None:
        study = OptimizationStudy(evaluator=evaluator, algorithm=ga(pop_size=8), n_gen=2, seed=0)
        with pytest.raises(OptimizationError):
            study.resume()


# --------------------------------------------------------------------------- #
# HistoryCallback
# --------------------------------------------------------------------------- #
class TestHistoryCallback:
    def test_empty_history_returns_empty_trace(self) -> None:
        cb = HistoryCallback()
        assert len(cb) == 0
        assert cb.best_per_generation().size == 0
