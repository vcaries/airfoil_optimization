"""Unit tests for convergence strategies and the chain-of-responsibility pipeline.

A :class:`FakeSolver` stands in for :class:`XfoilRunner` so we can exercise the
strategies without ever touching the binary or even subprocess.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from aeroforge.core.exceptions import ConvergenceError
from aeroforge.core.types import OperatingPoint
from aeroforge.geometry import Airfoil, NACA4Generator
from aeroforge.solver.base import AbstractSolver
from aeroforge.solver.convergence import (
    AlphaContinuationStrategy,
    ConvergencePipeline,
    IncreaseIterationsStrategy,
    InviscidInitStrategy,
    PerturbAlphaStrategy,
    RepanelStrategy,
)
from aeroforge.solver.xfoil.results import PolarPoint


# --------------------------------------------------------------------------- #
# Test double
# --------------------------------------------------------------------------- #
@dataclass
class FakeSolver(AbstractSolver):
    """A configurable stand-in for :class:`XfoilRunner`.

    The behaviour is driven by the ``behavior`` callable which inspects the
    current solver attributes (``max_iter``, ``repanel``) and the requested
    operating point, then either returns a :class:`PolarPoint` (convergence)
    or raises :class:`ConvergenceError` (non-convergence).
    """

    behavior: Callable[[FakeSolver, OperatingPoint], PolarPoint]
    max_iter: int = 200
    repanel: bool = False
    call_log: list[tuple[float, int, bool]] = field(default_factory=list)

    def analyze(self, airfoil: Airfoil, point: OperatingPoint) -> PolarPoint:  # noqa: D401
        """Delegate to the configured ``behavior`` callable."""
        self.call_log.append((point.alpha, self.max_iter, self.repanel))
        return self.behavior(self, point)


def _converged(point: OperatingPoint, cl: float = 0.5, cd: float = 0.01) -> PolarPoint:
    """Build a synthetic converged :class:`PolarPoint`."""
    return PolarPoint(
        operating_point=point,
        cl=cl,
        cd=cd,
        cdp=cd / 10,
        cm=-0.05,
        x_trans_upper=0.3,
        x_trans_lower=0.6,
        converged=True,
    )


@pytest.fixture
def airfoil() -> Airfoil:
    return NACA4Generator("0012", n_points=40).generate()


@pytest.fixture
def point() -> OperatingPoint:
    return OperatingPoint(alpha=5.0, reynolds=1.0e6)


# --------------------------------------------------------------------------- #
# IncreaseIterationsStrategy
# --------------------------------------------------------------------------- #
class TestIncreaseIterationsStrategy:
    def test_rejects_non_positive_factor(self) -> None:
        with pytest.raises(ValueError):
            IncreaseIterationsStrategy(factor=1.0)

    def test_succeeds_when_iter_cap_high_enough(
        self, airfoil: Airfoil, point: OperatingPoint
    ) -> None:
        def behavior(solver: FakeSolver, p: OperatingPoint) -> PolarPoint:
            if solver.max_iter < 300:
                raise ConvergenceError("still iterating", alpha=p.alpha)
            return _converged(p)

        solver = FakeSolver(behavior=behavior, max_iter=200)
        strategy = IncreaseIterationsStrategy(factor=2.0, max_iter=800)
        result = strategy.attempt(solver, airfoil, point, history=[])
        assert result.converged is True
        # The runner's max_iter was restored after the attempt.
        assert solver.max_iter == 200
        assert solver.call_log[-1][1] == 400  # the bumped iter

    def test_ceiling_blocks_retry(self, airfoil: Airfoil, point: OperatingPoint) -> None:
        solver = FakeSolver(
            behavior=lambda s, p: (_ for _ in ()).throw(ConvergenceError("nope", alpha=p.alpha)),
            max_iter=800,
        )
        strategy = IncreaseIterationsStrategy(factor=2.0, max_iter=800)
        with pytest.raises(ConvergenceError):
            strategy.attempt(solver, airfoil, point, history=[])


# --------------------------------------------------------------------------- #
# AlphaContinuationStrategy
# --------------------------------------------------------------------------- #
class TestAlphaContinuationStrategy:
    def test_rejects_non_positive_step(self) -> None:
        with pytest.raises(ValueError):
            AlphaContinuationStrategy(step=0.0)

    def test_requires_history(self, airfoil: Airfoil, point: OperatingPoint) -> None:
        solver = FakeSolver(behavior=lambda s, p: _converged(p))
        strategy = AlphaContinuationStrategy()
        with pytest.raises(ConvergenceError):
            strategy.attempt(solver, airfoil, point, history=[])

    def test_walks_alpha_in_small_steps(self, airfoil: Airfoil, point: OperatingPoint) -> None:
        # Anchor at alpha=4 deg, target alpha=5 deg, step=0.25 -> 4 intermediates.
        anchor = _converged(OperatingPoint(alpha=4.0, reynolds=1.0e6))
        solver = FakeSolver(behavior=lambda s, p: _converged(p))
        strategy = AlphaContinuationStrategy(step=0.25, max_steps=20)
        result = strategy.attempt(solver, airfoil, point, history=[anchor])
        assert result.operating_point.alpha == pytest.approx(5.0)
        # 4 steps of 0.25 deg from 4 -> 5.
        assert len(solver.call_log) == 4
        # Final alpha is the requested value, not an overshoot.
        assert solver.call_log[-1][0] == pytest.approx(5.0)

    def test_max_steps_blocks_excessive_walk(self, airfoil: Airfoil, point: OperatingPoint) -> None:
        # Anchor at -10 deg, target 5 deg, step=0.1 -> 150 steps > max_steps=10.
        anchor = _converged(OperatingPoint(alpha=-10.0, reynolds=1.0e6))
        solver = FakeSolver(behavior=lambda s, p: _converged(p))
        strategy = AlphaContinuationStrategy(step=0.1, max_steps=10)
        with pytest.raises(ConvergenceError):
            strategy.attempt(solver, airfoil, point, history=[anchor])


# --------------------------------------------------------------------------- #
# PerturbAlphaStrategy
# --------------------------------------------------------------------------- #
class TestPerturbAlphaStrategy:
    def test_succeeds_with_positive_perturbation(
        self, airfoil: Airfoil, point: OperatingPoint
    ) -> None:
        def behavior(solver: FakeSolver, p: OperatingPoint) -> PolarPoint:
            # Only accept alpha that differs from the original by exactly +0.05.
            if p.alpha == pytest.approx(5.05):
                return _converged(p)
            raise ConvergenceError("not yet", alpha=p.alpha)

        solver = FakeSolver(behavior=behavior)
        strategy = PerturbAlphaStrategy(epsilon=0.05)
        result = strategy.attempt(solver, airfoil, point, history=[])
        assert result.operating_point.alpha == pytest.approx(5.05)

    def test_both_perturbations_failing_raises(
        self, airfoil: Airfoil, point: OperatingPoint
    ) -> None:
        solver = FakeSolver(
            behavior=lambda s, p: (_ for _ in ()).throw(ConvergenceError("nope", alpha=p.alpha))
        )
        strategy = PerturbAlphaStrategy()
        with pytest.raises(ConvergenceError):
            strategy.attempt(solver, airfoil, point, history=[])


# --------------------------------------------------------------------------- #
# RepanelStrategy
# --------------------------------------------------------------------------- #
class TestRepanelStrategy:
    def test_enables_repanel_for_the_call(self, airfoil: Airfoil, point: OperatingPoint) -> None:
        def behavior(solver: FakeSolver, p: OperatingPoint) -> PolarPoint:
            if not solver.repanel:
                raise ConvergenceError("need repanel", alpha=p.alpha)
            return _converged(p)

        solver = FakeSolver(behavior=behavior, repanel=False)
        strategy = RepanelStrategy()
        result = strategy.attempt(solver, airfoil, point, history=[])
        assert result.converged
        # Original repanel attribute restored after the strategy.
        assert solver.repanel is False


# --------------------------------------------------------------------------- #
# InviscidInitStrategy
# --------------------------------------------------------------------------- #
class TestInviscidInitStrategy:
    def test_bumps_iter_for_the_retry(self, airfoil: Airfoil, point: OperatingPoint) -> None:
        def behavior(solver: FakeSolver, p: OperatingPoint) -> PolarPoint:
            if solver.max_iter < 400:
                raise ConvergenceError("not enough iter", alpha=p.alpha)
            return _converged(p)

        solver = FakeSolver(behavior=behavior, max_iter=200)
        strategy = InviscidInitStrategy(max_iter_bump=400)
        result = strategy.attempt(solver, airfoil, point, history=[])
        assert result.converged
        # The bumped iter value was used during the call.
        assert solver.call_log[-1][1] == 400
        # And the original was restored after the strategy.
        assert solver.max_iter == 200


# --------------------------------------------------------------------------- #
# ConvergencePipeline
# --------------------------------------------------------------------------- #
class TestConvergencePipeline:
    def test_rejects_empty_strategies(self) -> None:
        with pytest.raises(ValueError):
            ConvergencePipeline([])

    def test_returns_first_success(self, airfoil: Airfoil, point: OperatingPoint) -> None:
        def behavior(solver: FakeSolver, p: OperatingPoint) -> PolarPoint:
            # Only succeed when max_iter has been bumped by IncreaseIterations.
            if solver.max_iter >= 400:
                return _converged(p)
            raise ConvergenceError("nope", alpha=p.alpha)

        solver = FakeSolver(behavior=behavior, max_iter=200)
        pipeline = ConvergencePipeline(
            [
                IncreaseIterationsStrategy(factor=2.0, max_iter=800),
                PerturbAlphaStrategy(),
            ]
        )
        result = pipeline.attempt(solver, airfoil, point, history=[])
        assert result.converged

    def test_all_strategies_failing_raises(self, airfoil: Airfoil, point: OperatingPoint) -> None:
        solver = FakeSolver(
            behavior=lambda s, p: (_ for _ in ()).throw(ConvergenceError("nope", alpha=p.alpha)),
            max_iter=800,
        )
        pipeline = ConvergencePipeline(
            [
                IncreaseIterationsStrategy(factor=2.0, max_iter=800),
                PerturbAlphaStrategy(),
            ]
        )
        with pytest.raises(ConvergenceError) as info:
            pipeline.attempt(solver, airfoil, point, history=[])
        assert info.value.alpha == pytest.approx(point.alpha)
