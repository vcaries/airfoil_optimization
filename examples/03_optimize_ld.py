"""Single-objective L/D maximisation over a NACA 4-digit design space.

A tiny GA run (pop=12, n_gen=8) demonstrating the full optimisation chain:
genome -> NACA factory -> SyntheticSolver -> objective. The synthetic solver
is concave so the GA converges visibly in a few generations.

Replace ``SyntheticSolver`` with :class:`aeroforge.solver.XfoilRunner` to drive
the real binary.

Run::

    python examples/03_optimize_ld.py
"""

from __future__ import annotations

from dataclasses import dataclass

from aeroforge.core.types import OperatingPoint
from aeroforge.geometry import Airfoil, NACA4Generator
from aeroforge.optimization import (
    AirfoilEvaluator,
    DesignSpace,
    DesignVariable,
    MaximizeLiftToDrag,
    MinThicknessConstraint,
    OptimizationStudy,
)
from aeroforge.optimization.algorithms import ga
from aeroforge.solver.base import AbstractSolver
from aeroforge.solver.xfoil.results import PolarPoint


@dataclass
class SyntheticSolver(AbstractSolver):
    """Peaked synthetic L/D so the GA has a clear sweet spot to converge to."""

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


def build_naca4(params: dict[str, float]) -> Airfoil:
    """Decode a 3-vector genome into a NACA 4-digit airfoil."""
    m = int(round(params["m"] * 9))
    p_digit = max(int(round(params["p"] * 9)), 1)
    t_pct = max(int(round(params["t"] * 24)) + 6, 6)
    return NACA4Generator(f"{m}{p_digit}{t_pct:02d}", n_points=80).generate()


def main() -> None:
    """Set up the evaluator, launch the GA, print the best design."""
    design_space = DesignSpace(
        [
            DesignVariable("m", 0.0, 0.9, label="camber digit / 10"),
            DesignVariable("p", 0.1, 1.0, label="camber-pos digit / 10"),
            DesignVariable("t", 0.0, 1.0, label="(thickness - 6) / 24"),
        ]
    )

    evaluator = AirfoilEvaluator(
        design_space=design_space,
        airfoil_factory=build_naca4,
        solver=SyntheticSolver(),
        operating_point=OperatingPoint(alpha=4.0, reynolds=5.0e5),
        objectives=[MaximizeLiftToDrag()],
        geometric_constraints=[MinThicknessConstraint(t_min=0.08)],
    )

    study = OptimizationStudy(evaluator=evaluator, algorithm=ga(pop_size=12), n_gen=8, seed=42)
    print("Running GA (pop=12, n_gen=8, seed=42) ...")
    result = study.run()

    best = evaluator.genome_to_airfoil(result.X)
    best_ld = -float(result.F[0])
    print(f"\nBest design: {best.name}")
    print(f"  L/D = {best_ld:.2f}")
    print(f"  max thickness = {best.max_thickness:.4f}")
    print(f"  max camber    = {best.max_camber:.4f}")
    print(f"  generations captured: {len(study.history)}")


if __name__ == "__main__":
    main()
