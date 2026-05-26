"""Compute a viscous polar of a NACA 2412 from alpha = -2 to alpha = 12 deg.

Requires the XFOIL binary on PATH. Planned for milestone M2 -- this script
is included so the intended end-to-end shape of the API is visible up front.

Run (once the solver is implemented)::

    python examples/02_run_polar.py
"""

from __future__ import annotations

from aeroforge.core.types import OperatingPoint
from aeroforge.geometry import NACA4Generator
from aeroforge.solver import XfoilRunner, XfoilSession


def main() -> None:
    """Build the session, run XFOIL, and print the resulting polar."""
    airfoil = NACA4Generator("2412", n_points=120).generate()

    alphas = [a * 0.5 for a in range(-4, 25)]  # -2 .. 12 deg, step 0.5
    points = [OperatingPoint(alpha=a, reynolds=1.0e6) for a in alphas]
    session = XfoilSession(airfoil=airfoil, operating_points=points, max_iter=300)

    solver = XfoilRunner(binary="xfoil")
    # The full execution path is implemented in milestone M2; for now this
    # script documents the intended API shape end to end.
    _ = session.to_command_script(dat_path="naca2412.dat", polar_path="polar.pol")
    print(f"Solver: {solver.binary}")
    print(f"Built XFOIL session: {len(session.operating_points)} operating points.")


if __name__ == "__main__":  # pragma: no cover
    main()
