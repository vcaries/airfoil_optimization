"""Unit tests for the campaigns layer (ResultStore + CampaignRunner)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from aeroforge.campaigns import (
    CampaignRunner,
    SqliteResultStore,
    Sweep,
    hash_airfoil,
)
from aeroforge.campaigns.runner import CampaignResult
from aeroforge.core.exceptions import ConvergenceError
from aeroforge.core.types import OperatingPoint
from aeroforge.geometry import Airfoil, NACA4Generator
from aeroforge.solver.base import AbstractSolver
from aeroforge.solver.convergence import (
    ConvergencePipeline,
    IncreaseIterationsStrategy,
)
from aeroforge.solver.xfoil.results import PolarPoint


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
@dataclass
class CountingSolver(AbstractSolver):
    """FakeSolver that records every analyze() call.

    The synthetic CL is alpha-dependent so cached vs fresh results are
    distinguishable by value; the call counter lets us assert caching.
    """

    fail_at: set[float] = field(default_factory=set)
    max_iter: int = 200
    repanel: bool = False
    calls: int = 0

    def analyze(self, airfoil: Airfoil, point: OperatingPoint) -> PolarPoint:
        self.calls += 1
        if point.alpha in self.fail_at:
            raise ConvergenceError("synthetic failure", alpha=point.alpha)
        return PolarPoint(
            operating_point=point,
            cl=0.5 + 0.1 * point.alpha,
            cd=0.008,
            cdp=0.001,
            cm=-0.05,
            x_trans_upper=0.3,
            x_trans_lower=0.6,
            converged=True,
        )


@pytest.fixture
def airfoil() -> Airfoil:
    return NACA4Generator("0012", n_points=40).generate()


@pytest.fixture
def sweep(airfoil: Airfoil) -> Sweep:
    """A 5-alpha sweep at Re = 1e6."""
    return Sweep(
        parameters={"alpha": [0.0, 2.0, 4.0, 6.0, 8.0]},
        factory=lambda p: (airfoil, OperatingPoint(alpha=p["alpha"], reynolds=1.0e6)),
    )


# --------------------------------------------------------------------------- #
# hash_airfoil
# --------------------------------------------------------------------------- #
class TestHashAirfoil:
    def test_is_deterministic(self, airfoil: Airfoil) -> None:
        assert hash_airfoil(airfoil) == hash_airfoil(airfoil)

    def test_distinguishes_different_airfoils(self) -> None:
        a = NACA4Generator("0012", n_points=40).generate()
        b = NACA4Generator("2412", n_points=40).generate()
        assert hash_airfoil(a) != hash_airfoil(b)

    def test_returns_16_hex_chars(self, airfoil: Airfoil) -> None:
        h = hash_airfoil(airfoil)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


# --------------------------------------------------------------------------- #
# SqliteResultStore
# --------------------------------------------------------------------------- #
class TestSqliteResultStore:
    def _sample_point(self, alpha: float = 2.0) -> PolarPoint:
        return PolarPoint(
            operating_point=OperatingPoint(alpha=alpha, reynolds=1e6),
            cl=0.5,
            cd=0.008,
            cdp=0.001,
            cm=-0.05,
            x_trans_upper=0.3,
            x_trans_lower=0.6,
            converged=True,
        )

    def test_round_trip(self, tmp_path: Path) -> None:
        store = SqliteResultStore(tmp_path / "store.db")
        pt = self._sample_point()
        store.write("abc123", pt)
        assert store.has("abc123", pt.operating_point)
        loaded = store.load("abc123", pt.operating_point)
        assert loaded is not None
        assert loaded.cl == pytest.approx(0.5)
        assert loaded.converged is True
        store.close()

    def test_has_returns_false_for_missing(self, tmp_path: Path) -> None:
        store = SqliteResultStore(tmp_path / "store.db")
        assert not store.has("nope", OperatingPoint(alpha=2.0))
        store.close()

    def test_load_returns_none_for_missing(self, tmp_path: Path) -> None:
        store = SqliteResultStore(tmp_path / "store.db")
        assert store.load("nope", OperatingPoint(alpha=2.0)) is None
        store.close()

    def test_write_overwrites_existing_row(self, tmp_path: Path) -> None:
        store = SqliteResultStore(tmp_path / "store.db")
        first = self._sample_point()
        store.write("abc", first)
        # Same key, different CL: must replace.
        second = PolarPoint(
            operating_point=first.operating_point,
            cl=0.9,
            cd=0.005,
            cdp=0.001,
            cm=-0.05,
            x_trans_upper=0.3,
            x_trans_lower=0.6,
            converged=True,
        )
        store.write("abc", second)
        loaded = store.load("abc", first.operating_point)
        assert loaded is not None
        assert loaded.cl == pytest.approx(0.9)
        store.close()

    def test_load_all(self, tmp_path: Path) -> None:
        store = SqliteResultStore(tmp_path / "store.db")
        for a in [0.0, 2.0, 4.0]:
            store.write("hash1", self._sample_point(alpha=a))
        assert len(store.load_all()) == 3
        store.close()

    def test_context_manager_closes(self, tmp_path: Path) -> None:
        with SqliteResultStore(tmp_path / "store.db") as store:
            store.write("hash", self._sample_point())
        # Reopen and verify the row persisted across context-manager exit.
        with SqliteResultStore(tmp_path / "store.db") as store:
            assert len(store.load_all()) == 1

    def test_survives_restart(self, tmp_path: Path) -> None:
        db = tmp_path / "store.db"
        store = SqliteResultStore(db)
        store.write("h", self._sample_point())
        store.close()
        # New connection, same file.
        store2 = SqliteResultStore(db)
        assert store2.has("h", OperatingPoint(alpha=2.0, reynolds=1e6))
        store2.close()


# --------------------------------------------------------------------------- #
# CampaignRunner (serial)
# --------------------------------------------------------------------------- #
class TestCampaignRunnerSerial:
    def test_happy_path(self, sweep: Sweep) -> None:
        solver = CountingSolver()
        result = CampaignRunner(solver=solver).run(sweep)
        assert len(result.converged) == 5
        assert len(result.failed) == 0
        assert result.success_rate == pytest.approx(1.0)
        assert solver.calls == 5

    def test_persists_to_store(self, sweep: Sweep, tmp_path: Path) -> None:
        store = SqliteResultStore(tmp_path / "store.db")
        runner = CampaignRunner(solver=CountingSolver(), store=store)
        result = runner.run(sweep)
        assert len(result.converged) == 5
        assert len(store.load_all()) == 5
        store.close()

    def test_resume_skips_cached_points(self, sweep: Sweep, tmp_path: Path) -> None:
        """Re-running the same sweep against the same store skips the solver."""
        store_path = tmp_path / "store.db"

        # First run populates the store.
        s1 = CountingSolver()
        with SqliteResultStore(store_path) as store:
            CampaignRunner(solver=s1, store=store).run(sweep)
        assert s1.calls == 5

        # Second run sees a fresh solver — it should NOT be invoked at all.
        s2 = CountingSolver()
        with SqliteResultStore(store_path) as store:
            result = CampaignRunner(solver=s2, store=store).run(sweep)
        assert s2.calls == 0
        assert len(result.converged) == 5

    def test_records_failures(self, sweep: Sweep) -> None:
        """ConvergenceError without a pipeline is a soft failure."""
        # Fail at alpha=4 and alpha=6.
        solver = CountingSolver(fail_at={4.0, 6.0})
        result = CampaignRunner(solver=solver).run(sweep)
        assert len(result.converged) == 3
        assert len(result.failed) == 2
        # Failed points keep their requested alpha.
        failed_alphas = sorted(pt.alpha for _h, pt in result.failed)
        assert failed_alphas == [4.0, 6.0]

    def test_convergence_pipeline_rescues_failures(self, sweep: Sweep) -> None:
        """A pipeline that bumps max_iter and retries should rescue the run."""

        class FlakyCountingSolver(CountingSolver):
            """Fails when max_iter < 400; succeeds otherwise."""

            def analyze(self, airfoil, point):
                self.calls += 1
                if self.max_iter < 400:
                    raise ConvergenceError("needs more iter", alpha=point.alpha)
                return PolarPoint(
                    operating_point=point,
                    cl=0.5 + 0.1 * point.alpha,
                    cd=0.008,
                    cdp=0.001,
                    cm=-0.05,
                    x_trans_upper=0.3,
                    x_trans_lower=0.6,
                    converged=True,
                )

        pipeline = ConvergencePipeline([IncreaseIterationsStrategy(factor=2.0, max_iter=800)])
        result = CampaignRunner(solver=FlakyCountingSolver(), convergence=pipeline).run(sweep)
        # The pipeline rescues every alpha.
        assert len(result.converged) == 5
        assert len(result.failed) == 0


# --------------------------------------------------------------------------- #
# CampaignResult
# --------------------------------------------------------------------------- #
class TestCampaignResult:
    def test_empty_success_rate_is_zero(self) -> None:
        assert CampaignResult().success_rate == 0.0

    def test_success_rate_math(self) -> None:
        pt = PolarPoint(
            operating_point=OperatingPoint(alpha=0.0),
            cl=0.0,
            cd=0.005,
            cdp=0.001,
            cm=0.0,
            x_trans_upper=1.0,
            x_trans_lower=1.0,
            converged=True,
        )
        result = CampaignResult(
            converged=[pt, pt, pt],
            failed=[("h", OperatingPoint(alpha=10.0))],
        )
        assert result.success_rate == pytest.approx(0.75)
