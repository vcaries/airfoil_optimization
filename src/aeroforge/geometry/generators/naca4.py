"""NACA 4-digit airfoil generator.

Implements the classic NACA 4-digit series. The designation ``MPXX`` encodes:

* ``M`` -- maximum camber as a percentage of chord (first digit),
* ``P`` -- chordwise position of maximum camber in tenths of chord (second
  digit),
* ``XX`` -- maximum thickness as a percentage of chord (last two digits).

For example ``2412`` is 2% camber at 40% chord with 12% thickness, and ``0012``
is a symmetric 12%-thick section.

References:
    Abbott, I. H. & von Doenhoff, A. E., *Theory of Wing Sections*, 1959.
"""

from __future__ import annotations

import numpy as np

from aeroforge.core.exceptions import GeneratorError
from aeroforge.core.types import FloatArray, TrailingEdge
from aeroforge.geometry.airfoil import Airfoil
from aeroforge.geometry.generators.base import AirfoilGenerator
from aeroforge.geometry.operations.discretize import cosine_spacing

# Thickness-distribution polynomial coefficients (Abbott & von Doenhoff).
_A0 = 0.2969
_A1 = -0.1260
_A2 = -0.3516
_A3 = 0.2843
_A4_OPEN = -0.1015  # classic finite (open) trailing edge
_A4_CLOSED = -0.1036  # modified to close the trailing edge exactly


class NACA4Generator(AirfoilGenerator):
    """Generate a NACA 4-digit airfoil from its designation.

    Args:
        designation: The four-character NACA code, e.g. ``"2412"`` or ``"0012"``.
        n_points: Number of cosine-spaced chordwise stations **per surface**.
            The resulting contour has ``2 * n_points - 1`` points because the
            leading-edge node is shared between the two surfaces.
        trailing_edge: Whether to use the open (classic) or closed thickness
            polynomial. Defaults to :attr:`TrailingEdge.CLOSED`.

    Raises:
        GeneratorError: If the designation is malformed or the parameters are
            non-physical (e.g. nonzero camber with camber position at the LE).

    Example:
        >>> gen = NACA4Generator("2412", n_points=120)
        >>> airfoil = gen.generate()
        >>> round(airfoil.max_thickness, 3)
        0.12
    """

    def __init__(
        self,
        designation: str,
        n_points: int = 100,
        trailing_edge: TrailingEdge = TrailingEdge.CLOSED,
    ) -> None:
        """Initialize and validate the generator parameters."""
        self.designation = str(designation).strip()
        self.n_points = int(n_points)
        self.trailing_edge = TrailingEdge(trailing_edge)
        self._max_camber, self._camber_pos, self._thickness = self._parse(self.designation)
        if self.n_points < 2:
            raise GeneratorError(f"n_points must be >= 2, got {self.n_points}.")
        if self._max_camber > 0.0 and self._camber_pos <= 0.0:
            raise GeneratorError("Camber position (2nd digit) must be > 0 when camber is nonzero.")

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse(designation: str) -> tuple[float, float, float]:
        """Decode an ``MPXX`` designation into physical fractions.

        Args:
            designation: The four-character code.

        Returns:
            A tuple ``(max_camber, camber_position, thickness)`` as fractions
            of chord.

        Raises:
            GeneratorError: If the code is not four digits.
        """
        if len(designation) != 4 or not designation.isdigit():
            raise GeneratorError(f"NACA 4-digit designation must be 4 digits, got {designation!r}.")
        max_camber = int(designation[0]) / 100.0
        camber_pos = int(designation[1]) / 10.0
        thickness = int(designation[2:4]) / 100.0
        return max_camber, camber_pos, thickness

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        """str: The airfoil name, e.g. ``"NACA 2412"``."""
        return f"NACA {self.designation}"

    def generate(self) -> Airfoil:
        """Build the NACA 4-digit airfoil.

        Returns:
            A new :class:`Airfoil` with ``2 * n_points - 1`` points in Selig
            order (trailing edge -> upper -> leading edge -> lower -> trailing
            edge).
        """
        xc = cosine_spacing(self.n_points, 0.0, 1.0)
        yt = self._thickness_distribution(xc)
        yc, dyc_dx = self._camber_line(xc)
        theta = np.arctan(dyc_dx)

        # Offset the thickness perpendicular to the camber line.
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        x_upper = xc - yt * sin_t
        y_upper = yc + yt * cos_t
        x_lower = xc + yt * sin_t
        y_lower = yc - yt * cos_t

        # Assemble in Selig order: reverse the upper surface (TE -> LE) then
        # append the lower surface (LE -> TE), dropping the duplicated LE node.
        x_sel = np.concatenate([x_upper[::-1], x_lower[1:]])
        y_sel = np.concatenate([y_upper[::-1], y_lower[1:]])
        return Airfoil(x_sel, y_sel, name=self.name)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _thickness_distribution(self, xc: FloatArray) -> FloatArray:
        """Compute the half-thickness distribution ``yt(x)``.

        Args:
            xc: Chordwise stations in ``[0, 1]``.

        Returns:
            The half-thickness at each station.
        """
        a4 = _A4_CLOSED if self.trailing_edge is TrailingEdge.CLOSED else _A4_OPEN
        return (
            5.0
            * self._thickness
            * (_A0 * np.sqrt(xc) + _A1 * xc + _A2 * xc**2 + _A3 * xc**3 + a4 * xc**4)
        )

    def _camber_line(self, xc: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Compute the camber line ``yc(x)`` and its slope ``dyc/dx``.

        Args:
            xc: Chordwise stations in ``[0, 1]``.

        Returns:
            A tuple ``(yc, dyc_dx)`` evaluated at each station.
        """
        m, p = self._max_camber, self._camber_pos
        if m == 0.0:
            zeros = np.zeros_like(xc)
            return zeros, zeros

        yc = np.empty_like(xc)
        dyc = np.empty_like(xc)
        fore = xc < p  # ahead of max-camber location
        aft = ~fore

        yc[fore] = (m / p**2) * (2.0 * p * xc[fore] - xc[fore] ** 2)
        dyc[fore] = (2.0 * m / p**2) * (p - xc[fore])

        yc[aft] = (m / (1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * xc[aft] - xc[aft] ** 2)
        dyc[aft] = (2.0 * m / (1.0 - p) ** 2) * (p - xc[aft])
        return yc, dyc
