"""Solver layer: XFOIL wrapper and convergence-robustness pipeline."""

from aeroforge.solver.base import AbstractSolver
from aeroforge.solver.convergence import ConvergencePipeline, ConvergenceStrategy
from aeroforge.solver.xfoil import (
    CpDistribution,
    Polar,
    PolarPoint,
    XfoilCommand,
    XfoilOutputParser,
    XfoilRunner,
    XfoilSession,
)

__all__ = [
    "AbstractSolver",
    "XfoilRunner",
    "XfoilSession",
    "XfoilCommand",
    "XfoilOutputParser",
    "PolarPoint",
    "Polar",
    "CpDistribution",
    "ConvergenceStrategy",
    "ConvergencePipeline",
]
