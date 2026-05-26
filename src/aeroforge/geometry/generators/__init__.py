"""Airfoil generators (Strategy pattern).

All concrete generators implement :class:`AirfoilGenerator` and produce an
:class:`~aeroforge.geometry.airfoil.Airfoil`. Use them interchangeably as
shape sources in campaigns and optimizations.
"""

from aeroforge.geometry.generators.base import AirfoilGenerator
from aeroforge.geometry.generators.from_file import DatFileGenerator
from aeroforge.geometry.generators.naca4 import NACA4Generator
from aeroforge.geometry.generators.parametric import (
    BezierGenerator,
    CSTGenerator,
    PARSECGenerator,
)

__all__ = [
    "AirfoilGenerator",
    "NACA4Generator",
    "DatFileGenerator",
    "CSTGenerator",
    "BezierGenerator",
    "PARSECGenerator",
]
