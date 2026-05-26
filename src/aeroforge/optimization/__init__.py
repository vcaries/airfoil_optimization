"""Optimization layer (pymoo-based).

Importing this subpackage requires the ``optim`` extra
(``pip install aeroforge[optim]``). It is intentionally not imported by the
top-level :mod:`aeroforge` package so the core library can be used in
pymoo-free environments.
"""

from aeroforge.optimization.callbacks import GenerationSnapshot, HistoryCallback
from aeroforge.optimization.constraints import (
    GeometricConstraint,
    MinPitchingMomentConstraint,
    MinThicknessConstraint,
    PhysicalConstraint,
)
from aeroforge.optimization.evaluator import AirfoilEvaluator, AirfoilFactory
from aeroforge.optimization.objectives import (
    MaximizeLift,
    MaximizeLiftToDrag,
    MinimizeDrag,
    Objective,
)
from aeroforge.optimization.penalties import (
    exponential_penalty,
    linear_penalty,
    quadratic_penalty,
)
from aeroforge.optimization.problem import AirfoilProblem
from aeroforge.optimization.study import OptimizationStudy
from aeroforge.optimization.variables import DesignSpace, DesignVariable

__all__ = [
    # variables
    "DesignVariable",
    "DesignSpace",
    # objectives
    "Objective",
    "MinimizeDrag",
    "MaximizeLift",
    "MaximizeLiftToDrag",
    # constraints
    "GeometricConstraint",
    "PhysicalConstraint",
    "MinThicknessConstraint",
    "MinPitchingMomentConstraint",
    # penalties
    "quadratic_penalty",
    "linear_penalty",
    "exponential_penalty",
    # plumbing
    "AirfoilEvaluator",
    "AirfoilFactory",
    "AirfoilProblem",
    "OptimizationStudy",
    "HistoryCallback",
    "GenerationSnapshot",
]
