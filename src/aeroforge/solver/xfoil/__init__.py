"""XFOIL solver subsystem.

Layered design (top down):

* :class:`XfoilRunner` -- the public :class:`AbstractSolver` implementation
  that owns the subprocess lifecycle.
* :class:`XfoilSession` -- a declarative description of one XFOIL run.
* :class:`XfoilCommand` -- a fluent builder for XFOIL stdin command scripts.
* :class:`XfoilOutputParser` -- pure parsers for polar and Cp output files.
* Result dataclasses :class:`PolarPoint`, :class:`Polar`, :class:`CpDistribution`.
"""

from aeroforge.solver.xfoil.commands import XfoilCommand
from aeroforge.solver.xfoil.parser import XfoilOutputParser
from aeroforge.solver.xfoil.results import CpDistribution, Polar, PolarPoint
from aeroforge.solver.xfoil.runner import XfoilRunner
from aeroforge.solver.xfoil.session import XfoilSession

__all__ = [
    "XfoilRunner",
    "XfoilSession",
    "XfoilCommand",
    "XfoilOutputParser",
    "PolarPoint",
    "Polar",
    "CpDistribution",
]
