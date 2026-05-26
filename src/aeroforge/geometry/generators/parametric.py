"""Parametric airfoil generators (CST, Bézier, PARSEC).

These shape parameterizations express an airfoil as a small vector of design
variables and are the workhorses of aerodynamic shape optimization. The classes
below pin down the public interface so the optimization layer can already be
written against them; full implementations are planned for the next milestones
(see ``ARCHITECTURE.md`` roadmap).
"""

from __future__ import annotations

import numpy as np

from aeroforge.core.exceptions import GeneratorError
from aeroforge.core.types import FloatArray
from aeroforge.geometry.airfoil import Airfoil
from aeroforge.geometry.generators.base import AirfoilGenerator


class CSTGenerator(AirfoilGenerator):
    """Class/Shape Transformation (Kulfan) parameterization.

    A CST airfoil is a class function (controlling the LE/TE shape) multiplied
    by a Bernstein-polynomial shape function whose coefficients are the design
    variables. The two surfaces are parameterized independently.

    Args:
        upper_weights: Bernstein coefficients for the upper surface.
        lower_weights: Bernstein coefficients for the lower surface.
        n_points: Number of cosine-spaced chordwise stations per surface.
        te_thickness: Trailing-edge half-gap (added to the shape function).

    Raises:
        GeneratorError: If the weight arrays are empty or mismatched in a way
            that the implementation does not support.
    """

    def __init__(
        self,
        upper_weights: FloatArray,
        lower_weights: FloatArray,
        *,
        n_points: int = 100,
        te_thickness: float = 0.0,
    ) -> None:
        """Store CST coefficients and discretization settings."""
        self.upper_weights = np.asarray(upper_weights, dtype=float)
        self.lower_weights = np.asarray(lower_weights, dtype=float)
        self.n_points = int(n_points)
        self.te_thickness = float(te_thickness)
        if self.upper_weights.size == 0 or self.lower_weights.size == 0:
            raise GeneratorError("CST weight vectors must be non-empty.")

    @property
    def name(self) -> str:
        """str: A label including the polynomial order on each surface."""
        return f"CST(n_up={self.upper_weights.size}, n_lo={self.lower_weights.size})"

    def generate(self) -> Airfoil:
        """Build the CST airfoil.

        Returns:
            A new :class:`Airfoil`.

        Raises:
            NotImplementedError: Planned for the next milestone (see roadmap).
        """
        raise NotImplementedError(
            "CSTGenerator.generate is planned (see ARCHITECTURE.md roadmap, M3)."
        )


class BezierGenerator(AirfoilGenerator):
    """Cubic-Bézier per-surface airfoil parameterization.

    Each surface is a chain of cubic Bézier segments whose control points are
    the design variables. Convenient for human-in-the-loop shape editing.

    Args:
        upper_control: ``(n, 2)`` array of upper-surface control points.
        lower_control: ``(n, 2)`` array of lower-surface control points.
        n_points: Number of evaluation points per surface.
    """

    def __init__(
        self,
        upper_control: FloatArray,
        lower_control: FloatArray,
        *,
        n_points: int = 100,
    ) -> None:
        """Store Bézier control polygons."""
        self.upper_control = np.asarray(upper_control, dtype=float)
        self.lower_control = np.asarray(lower_control, dtype=float)
        self.n_points = int(n_points)

    def generate(self) -> Airfoil:
        """Build the Bézier airfoil.

        Returns:
            A new :class:`Airfoil`.

        Raises:
            NotImplementedError: Planned for the next milestone (see roadmap).
        """
        raise NotImplementedError(
            "BezierGenerator.generate is planned (see ARCHITECTURE.md roadmap, M3)."
        )


class PARSECGenerator(AirfoilGenerator):
    """PARSEC 11-parameter airfoil parameterization.

    Sobieczky's PARSEC family uses 11 physically meaningful parameters (LE
    radius, crest position, crest curvature, TE angles, ...) which makes it
    ideal for constrained aerodynamic optimization.

    Args:
        parameters: Mapping from PARSEC parameter name to value. Expected keys
            include ``r_le``, ``x_up``, ``z_up``, ``z_xx_up``, ``x_lo``,
            ``z_lo``, ``z_xx_lo``, ``z_te``, ``dz_te``, ``alpha_te``,
            ``beta_te``.
        n_points: Number of cosine-spaced stations per surface.
    """

    def __init__(self, parameters: dict[str, float], *, n_points: int = 100) -> None:
        """Store the 11 PARSEC parameters."""
        self.parameters = dict(parameters)
        self.n_points = int(n_points)

    def generate(self) -> Airfoil:
        """Build the PARSEC airfoil.

        Returns:
            A new :class:`Airfoil`.

        Raises:
            NotImplementedError: Planned for the next milestone (see roadmap).
        """
        raise NotImplementedError(
            "PARSECGenerator.generate is planned (see ARCHITECTURE.md roadmap, M3)."
        )
