"""Compute a viscous polar of a NACA 2412 from alpha = -2 to alpha = 12 deg.

Uses the real XFOIL binary if it is on PATH, otherwise falls back to a small
synthetic solver so the script always runs to completion (handy on CI and on
machines without XFOIL installed).

Run::

    python examples/02_run_polar.py                 # auto-detect XFOIL
    python examples/02_run_polar.py --synthetic     # force synthetic
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass

from aeroforge.core.types import OperatingPoint
from aeroforge.geometry import Airfoil, NACA4Generator
from aeroforge.solver.base import AbstractSolver
from aeroforge.solver.xfoil.results import PolarPoint


@dataclass
class SyntheticSolver(AbstractSolver):
    """A toy lift/drag model: CL linear in alpha, CD quadratic in alpha."""

    def analyze(self, airfoil: Airfoil, point: OperatingPoint) -> PolarPoint:
        cl = 0.10 * point.alpha + 5.0 * airfoil.max_camber
        cd = 0.005 + 0.0008 * point.alpha**2
        return PolarPoint(
            operating_point=point,
            cl=cl,
            cd=cd,
            cdp=cd / 8,
            cm=-0.05,
            x_trans_upper=0.3,
            x_trans_lower=0.6,
            converged=True,
        )


def main() -> None:
    """Build a polar sweep and print the resulting (alpha, CL, CD, L/D) table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="Force the synthetic solver.")
    args = parser.parse_args()

    airfoil = NACA4Generator("2412", n_points=120).generate()
    alphas = [a * 0.5 for a in range(-4, 25)]
    points = [OperatingPoint(alpha=a, reynolds=1.0e6) for a in alphas]

    if args.synthetic or shutil.which("xfoil") is None:
        if not args.synthetic:
            print("[info] xfoil not on PATH, using SyntheticSolver.")
        solver: AbstractSolver = SyntheticSolver()
    else:
        from aeroforge.solver import XfoilRunner

        solver = XfoilRunner("xfoil")
        print(f"[info] using XFOIL at {solver.binary}")

    print(f"\nPolar of {airfoil.name} at Re = 1.0e6:")
    print(f"  {'alpha':>6}  {'CL':>7}  {'CD':>8}  {'L/D':>7}")
    for point in points:
        try:
            result = solver.analyze(airfoil, point)
        except Exception as exc:  # noqa: BLE001
            print(f"  {point.alpha:>6.2f}  ---  ---  ---  ({exc})")
            continue
        print(
            f"  {result.operating_point.alpha:>6.2f}  {result.cl:>7.4f}  "
            f"{result.cd:>8.5f}  {result.lift_to_drag:>7.1f}"
        )


if __name__ == "__main__":
    main()
