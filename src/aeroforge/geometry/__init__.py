"""Airfoil geometry: representation, generation, and manipulation.

Public surface:

* :class:`Airfoil` -- value object holding Selig-ordered coordinates and
  exposing derived geometric quantities.
* :class:`AirfoilGenerator` and concrete subclasses -- Strategy pattern for
  producing airfoils from designations, parameters, or files.
* :mod:`aeroforge.geometry.operations` -- pure-function helpers for
  discretization, smoothing, and affine transforms.
* :mod:`aeroforge.geometry.metrics` -- pure-function geometric metrics
  (thickness, camber, area) that the :class:`Airfoil` class delegates to.
"""

from aeroforge.geometry.airfoil import Airfoil
from aeroforge.geometry.generators import (
    AirfoilGenerator,
    BezierGenerator,
    CSTGenerator,
    DatFileGenerator,
    NACA4Generator,
    PARSECGenerator,
)

__all__ = [
    "Airfoil",
    "AirfoilGenerator",
    "NACA4Generator",
    "DatFileGenerator",
    "CSTGenerator",
    "BezierGenerator",
    "PARSECGenerator",
]
