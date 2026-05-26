"""Generator that loads an airfoil from a user-supplied coordinate file.

Wrapping file loading in an :class:`AirfoilGenerator` means imported airfoils
are interchangeable with analytically generated ones everywhere downstream
(campaigns, optimization seeds, visualization).
"""

from __future__ import annotations

from pathlib import Path

from aeroforge.geometry.airfoil import Airfoil
from aeroforge.geometry.generators.base import AirfoilGenerator
from aeroforge.geometry.operations.discretize import repanel

PathLike = str | Path


class DatFileGenerator(AirfoilGenerator):
    """Load an airfoil from a Selig-format ``.dat`` file.

    Args:
        path: Path to the coordinate file.
        repanel_to: If given, resample the contour to this many points using a
            cosine arc-length distribution. Useful to normalize panel counts
            across a heterogeneous set of input airfoils.

    Example:
        >>> gen = DatFileGenerator("airfoils/e387.dat", repanel_to=160)
        >>> airfoil = gen.generate()  # doctest: +SKIP
    """

    def __init__(self, path: PathLike, *, repanel_to: int | None = None) -> None:
        """Store the file path and optional target panel count."""
        self.path = Path(path)
        self.repanel_to = repanel_to

    @property
    def name(self) -> str:
        """str: The airfoil name derived from the file stem."""
        return self.path.stem

    def generate(self) -> Airfoil:
        """Load (and optionally repanel) the airfoil.

        Returns:
            A new :class:`Airfoil`.

        Raises:
            InvalidAirfoilError: If the file cannot be parsed.
        """
        airfoil = Airfoil.from_dat(self.path)
        if self.repanel_to is not None:
            x, y = repanel(airfoil.x, airfoil.y, self.repanel_to)
            airfoil = Airfoil(x, y, name=airfoil.name)
        return airfoil
