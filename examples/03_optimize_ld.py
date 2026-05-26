"""Single-objective L/D maximisation over a NACA 4-digit design space.

Demonstrates the intended optimization API. The actual evaluation loop is
planned for milestone M3 -- this script shows the wiring users will write.

Requires the ``optim`` extra (``pip install aeroforge[optim]``).
"""

from __future__ import annotations

from aeroforge.core.types import OperatingPoint
from aeroforge.geometry import NACA4Generator


def build_naca4(params: dict[str, float]):
    """Build a NACA 4-digit airfoil from continuous design variables.

    Args:
        params: Mapping with keys ``m``, ``p``, ``t`` in [0, 1] fractions.

    Returns:
        A :class:`~aeroforge.geometry.airfoil.Airfoil`.
    """
    m = int(round(params["m"] * 9))  # 0..9 -> camber digit
    p = int(round(params["p"] * 9))  # 0..9 -> camber position digit
    t = int(round(params["t"] * 24)) + 6  # 6..30 -> thickness digits
    designation = f"{m}{p}{t:02d}"
    return NACA4Generator(designation, n_points=120).generate()


def main() -> None:
    """Set up the optimization wiring (full run requires milestone M3)."""
    from aeroforge.optimization import (
        AirfoilEvaluator,
        DesignSpace,
        DesignVariable,
        MaximizeLiftToDrag,
        MinThicknessConstraint,
    )

    design_space = DesignSpace(
        [
            DesignVariable("m", 0.0, 1.0, label="camber digit / 9"),
            DesignVariable("p", 0.1, 1.0, label="camber-pos digit / 9"),
            DesignVariable("t", 0.0, 1.0, label="(thickness - 6) / 24"),
        ]
    )

    # Evaluator wiring -- the actual solver call lives behind the M3 milestone.
    evaluator = AirfoilEvaluator(
        design_space=design_space,
        airfoil_factory=build_naca4,
        solver=None,  # type: ignore[arg-type]
        operating_point=OperatingPoint(alpha=4.0, reynolds=5.0e5),
        objectives=[MaximizeLiftToDrag()],
        geometric_constraints=[MinThicknessConstraint(t_min=0.09)],
    )
    print("Design space:", evaluator.design_space.names)
    print("Objectives  :", [o.name for o in evaluator.objectives])


if __name__ == "__main__":  # pragma: no cover
    main()
