"""Generate a NACA 2412 airfoil, print a few geometric metrics, and save it.

Run::

    python examples/01_generate_airfoil.py
"""

from __future__ import annotations

from aeroforge.geometry import NACA4Generator


def main() -> None:
    """Generate, inspect, and serialize the airfoil."""
    airfoil = NACA4Generator(designation="2412", n_points=120).generate()
    print(airfoil)
    print(f"  n_points     = {airfoil.n_points}")
    print(
        f"  max thickness = {airfoil.max_thickness:.4f}  at x/c = "
        f"{airfoil.max_thickness_location:.3f}"
    )
    print(
        f"  max camber    = {airfoil.max_camber:.4f}  at x/c = "
        f"{airfoil.max_camber_location:.3f}"
    )
    print(f"  enclosed area = {airfoil.area:.4f}")
    print(f"  TE gap        = {airfoil.trailing_edge_gap:.6f}")
    out = airfoil.to_dat("naca2412.dat")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
