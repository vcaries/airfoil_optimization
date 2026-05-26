"""High-level XFOIL session: airfoil + operating envelope + command script.

Wraps the low-level :class:`XfoilCommand` builder so callers think in terms of
aerodynamics ("analyse this airfoil at these alphas") rather than XFOIL menu
keystrokes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aeroforge.core.types import OperatingPoint
from aeroforge.solver.xfoil.commands import XfoilCommand

if TYPE_CHECKING:
    from aeroforge.geometry.airfoil import Airfoil


@dataclass(slots=True)
class XfoilSession:
    """A declarative description of one XFOIL run.

    Attributes:
        airfoil: The airfoil to load.
        operating_points: The aerodynamic states to evaluate.
        max_iter: Viscous iteration cap.
        n_crit: Transition amplification factor.
        repanel: Whether to call ``PANE`` after loading the airfoil.
    """

    airfoil: Airfoil
    operating_points: list[OperatingPoint] = field(default_factory=list)
    max_iter: int = 200
    n_crit: float = 9.0
    repanel: bool = True

    def to_command_script(self, *, dat_path: str, polar_path: str | None = None) -> str:
        """Render this session into an XFOIL stdin transcript.

        Args:
            dat_path: Path to the ``.dat`` file XFOIL should load.
            polar_path: Optional polar-accumulation output file.

        Returns:
            The newline-separated command transcript.

        Raises:
            ValueError: If :attr:`operating_points` is empty.
        """
        if not self.operating_points:
            raise ValueError("XfoilSession requires at least one operating point.")

        cmd = XfoilCommand().load(dat_path)
        if self.repanel:
            cmd.pane()
        cmd.oper().n_crit(self.n_crit).iter(self.max_iter)

        # Assume a homogeneous Re/Mach across all points (typical use case).
        first = self.operating_points[0]
        if first.is_viscous:
            cmd.viscous(reynolds=first.reynolds, mach=first.mach)

        if polar_path is not None:
            cmd.polar_accumulate(polar_path)

        for point in self.operating_points:
            cmd.alpha(point.alpha)

        if polar_path is not None:
            cmd.polar_close()

        return cmd.back().quit().build()
