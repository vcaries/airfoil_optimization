"""Unit tests for :class:`aeroforge.solver.xfoil.runner.XfoilRunner`.

These tests exercise the runner without requiring the real XFOIL binary by
monkey-patching :func:`shutil.which` (so construction succeeds) and
:func:`subprocess.run` (so :meth:`analyze` does not actually launch a process).
The mock writes a canned polar to the scratch directory in the same way the
real XFOIL would, then returns a fake :class:`subprocess.CompletedProcess`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from aeroforge.core.exceptions import (
    ConvergenceError,
    XfoilExecutionError,
    XfoilNotFoundError,
)
from aeroforge.core.types import OperatingPoint
from aeroforge.geometry import NACA4Generator
from aeroforge.solver.xfoil import runner as runner_mod
from aeroforge.solver.xfoil.runner import XfoilRunner

FIXTURE_POLAR = Path(__file__).resolve().parents[1] / "data" / "polars" / "naca0012_re1e6.pol"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_xfoil_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``shutil.which('xfoil')`` resolve to a fake path."""
    monkeypatch.setattr(runner_mod.shutil, "which", lambda name: "/fake/path/to/xfoil")


def _make_fake_run(
    *,
    polar_source: Path | None = FIXTURE_POLAR,
    returncode: int = 0,
    stderr: str = "",
    timeout: bool = False,
    raise_oserror: bool = False,
):
    """Build a fake :func:`subprocess.run` replacement.

    Args:
        polar_source: If given, the contents are copied to ``cwd/polar.pol``.
            Pass ``None`` to simulate XFOIL producing no polar file.
        returncode: Return code on the fake :class:`CompletedProcess`.
        stderr: Stderr text on the fake process.
        timeout: If ``True``, raise :class:`subprocess.TimeoutExpired`.
        raise_oserror: If ``True``, raise :class:`OSError`.
    """

    def _run(cmd, **kwargs: Any) -> subprocess.CompletedProcess:
        if timeout:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))
        if raise_oserror:
            raise OSError("simulated OS failure")
        cwd = Path(kwargs["cwd"])
        if polar_source is not None:
            (cwd / "polar.pol").write_text(
                polar_source.read_text(encoding="utf-8"), encoding="utf-8"
            )
        return subprocess.CompletedProcess(
            args=cmd, returncode=returncode, stdout="", stderr=stderr
        )

    return _run


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
class TestConstruction:
    """The constructor validates that the binary can be located."""

    def test_missing_binary_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runner_mod.shutil, "which", lambda name: None)
        with pytest.raises(XfoilNotFoundError):
            XfoilRunner("xfoil")

    def test_default_parameters(self, fake_xfoil_on_path: None) -> None:
        runner = XfoilRunner("xfoil")
        assert runner.max_iter == 200
        assert runner.n_crit == pytest.approx(9.0)
        assert runner.repanel is True
        assert runner.timeout_s == pytest.approx(60.0)


# --------------------------------------------------------------------------- #
# Successful runs
# --------------------------------------------------------------------------- #
class TestAnalyzeSuccess:
    """Happy-path runs return the right :class:`PolarPoint`."""

    @pytest.fixture
    def runner(self, fake_xfoil_on_path: None) -> XfoilRunner:
        return XfoilRunner("xfoil")

    def test_returns_matching_alpha(
        self,
        runner: XfoilRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(runner_mod.subprocess, "run", _make_fake_run())
        airfoil = NACA4Generator("0012", n_points=40).generate()
        point = OperatingPoint(alpha=2.0, reynolds=1.0e6)
        result = runner.analyze(airfoil, point)
        # Fixture row alpha=2: CL=0.2436, CD=0.00567.
        assert result.operating_point.alpha == pytest.approx(2.0)
        assert result.cl == pytest.approx(0.2436)
        assert result.cd == pytest.approx(0.00567)
        assert result.converged is True

    def test_alpha_not_in_polar_raises_convergence(
        self,
        runner: XfoilRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(runner_mod.subprocess, "run", _make_fake_run())
        airfoil = NACA4Generator("0012", n_points=40).generate()
        # alpha=20 is not in the fixture polar.
        point = OperatingPoint(alpha=20.0, reynolds=1.0e6)
        with pytest.raises(ConvergenceError) as info:
            runner.analyze(airfoil, point)
        assert info.value.alpha == pytest.approx(20.0)


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #
class TestAnalyzeFailures:
    """Process-level failures are translated to typed exceptions."""

    @pytest.fixture
    def runner(self, fake_xfoil_on_path: None) -> XfoilRunner:
        return XfoilRunner("xfoil", timeout_s=5.0)

    def test_timeout_is_translated(
        self,
        runner: XfoilRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(runner_mod.subprocess, "run", _make_fake_run(timeout=True))
        airfoil = NACA4Generator("0012", n_points=40).generate()
        with pytest.raises(XfoilExecutionError) as info:
            runner.analyze(airfoil, OperatingPoint(alpha=0.0, reynolds=1.0e6))
        assert "timed out" in str(info.value).lower()

    def test_non_zero_exit_is_translated(
        self,
        runner: XfoilRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            runner_mod.subprocess,
            "run",
            _make_fake_run(returncode=1, stderr="segfault!"),
        )
        airfoil = NACA4Generator("0012", n_points=40).generate()
        with pytest.raises(XfoilExecutionError) as info:
            runner.analyze(airfoil, OperatingPoint(alpha=0.0, reynolds=1.0e6))
        assert "exited with code 1" in str(info.value)

    def test_missing_polar_raises_convergence(
        self,
        runner: XfoilRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(runner_mod.subprocess, "run", _make_fake_run(polar_source=None))
        airfoil = NACA4Generator("0012", n_points=40).generate()
        with pytest.raises(ConvergenceError):
            runner.analyze(airfoil, OperatingPoint(alpha=0.0, reynolds=1.0e6))

    def test_oserror_is_translated(
        self,
        runner: XfoilRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(runner_mod.subprocess, "run", _make_fake_run(raise_oserror=True))
        airfoil = NACA4Generator("0012", n_points=40).generate()
        with pytest.raises(XfoilExecutionError):
            runner.analyze(airfoil, OperatingPoint(alpha=0.0, reynolds=1.0e6))
