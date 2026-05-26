"""Convergence-robustness strategies for the XFOIL wrapper."""

from aeroforge.solver.convergence.base import ConvergenceStrategy
from aeroforge.solver.convergence.pipeline import ConvergencePipeline
from aeroforge.solver.convergence.strategies import (
    AlphaContinuationStrategy,
    IncreaseIterationsStrategy,
    InviscidInitStrategy,
    PerturbAlphaStrategy,
    RepanelStrategy,
)

__all__ = [
    "ConvergenceStrategy",
    "ConvergencePipeline",
    "IncreaseIterationsStrategy",
    "AlphaContinuationStrategy",
    "InviscidInitStrategy",
    "RepanelStrategy",
    "PerturbAlphaStrategy",
]
