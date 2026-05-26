"""Smoke tests for the top-level :mod:`aeroforge` namespace."""

from __future__ import annotations


def test_top_level_import_succeeds() -> None:
    """Importing the package does not raise."""
    import aeroforge

    assert isinstance(aeroforge.__version__, str)


def test_public_symbols_are_re_exported() -> None:
    """The most-used classes are reachable from the top-level namespace."""
    import aeroforge

    for name in (
        "Airfoil",
        "AirfoilGenerator",
        "NACA4Generator",
        "DatFileGenerator",
        "OperatingPoint",
        "AeroforgeError",
        "ConvergenceError",
    ):
        assert hasattr(aeroforge, name), f"aeroforge.{name} is missing"


def test_solver_subpackage_imports() -> None:
    """The solver subpackage imports without needing the XFOIL binary."""
    import aeroforge.solver as solver  # noqa: PLC0415

    # Stable public surface for the wrapper.
    for name in (
        "AbstractSolver",
        "XfoilSession",
        "XfoilCommand",
        "ConvergencePipeline",
        "PolarPoint",
    ):
        assert hasattr(solver, name), f"aeroforge.solver.{name} is missing"


def test_xfoil_command_builder_round_trip() -> None:
    """The :class:`XfoilCommand` builder produces a parseable transcript."""
    from aeroforge.solver import XfoilCommand

    script = (
        XfoilCommand()
        .load("airfoil.dat")
        .oper()
        .viscous(reynolds=1.0e6, mach=0.0)
        .iter(200)
        .alpha(2.0)
        .quit()
        .build()
    )
    lines = script.splitlines()
    assert lines[0] == "LOAD"
    assert "VISC" in lines
    assert "ALFA" in lines
    assert lines[-1] == "QUIT"
