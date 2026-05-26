"""Abstract base class for airfoil generators (Strategy pattern).

Every way of producing an :class:`~aeroforge.geometry.airfoil.Airfoil` -- NACA
series, parametric families (CST/Bézier/PARSEC), or loading from file -- is a
concrete :class:`AirfoilGenerator`. The optimization layer programs against this
interface alone, so new shape parameterizations can be added without touching
the optimizer (open/closed principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aeroforge.geometry.airfoil import Airfoil


class AirfoilGenerator(ABC):
    """Interface for objects that produce an :class:`Airfoil`.

    Concrete subclasses store their shape parameters as instance state and
    return a fresh airfoil from :meth:`generate`. They should be cheap to
    construct and free of side effects.
    """

    @abstractmethod
    def generate(self) -> Airfoil:
        """Build and return the airfoil described by this generator.

        Returns:
            A new :class:`Airfoil` instance.

        Raises:
            GeneratorError: If the stored parameters are invalid.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """str: A label describing the generated airfoil.

        Subclasses should override this to produce a meaningful name (e.g.
        ``"NACA 2412"``). Defaults to the class name.
        """
        return type(self).__name__
