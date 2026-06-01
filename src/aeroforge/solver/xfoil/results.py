"""Typed result containers for XFOIL output.

Using plain dataclasses (not pandas) keeps the solver dependency-light and the
results trivially serializable. Aggregations into DataFrames happen at the
campaign / visualization layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aeroforge.core.types import FloatArray, OperatingPoint


@dataclass(frozen=True, slots=True)
class PolarPoint:
    """A single converged operating-point solution.

    Attributes:
        operating_point: The operating point this result corresponds to.
        cl: Lift coefficient.
        cd: Drag coefficient.
        cdp: Pressure component of the drag coefficient.
        cm: Pitching-moment coefficient (about the quarter-chord).
        x_trans_upper: Upper-surface transition location (x/c).
        x_trans_lower: Lower-surface transition location (x/c).
        converged: Whether the solver flagged this point as converged.
    """

    operating_point: OperatingPoint
    cl: float
    cd: float
    cdp: float
    cm: float
    x_trans_upper: float
    x_trans_lower: float
    converged: bool = True

    @property
    def lift_to_drag(self) -> float:
        """float: Lift-to-drag ratio (returns ``inf`` if drag is zero)."""
        return self.cl / self.cd if self.cd != 0.0 else float("inf")


@dataclass(frozen=True, slots=True)
class CpDistribution:
    """Chordwise pressure-coefficient distribution.

    Attributes:
        operating_point: The operating point this distribution was sampled at.
        x: Chordwise stations.
        cp: Pressure coefficients matching ``x``.
    """

    operating_point: OperatingPoint
    x: FloatArray
    cp: FloatArray


@dataclass(frozen=True, slots=True)
class WallProfile:
    """Boundary-layer surface distributions from XFOIL's ``DUMP`` command.

    The eight columns mirror XFOIL's ``DUMP`` output format
    (``s x y Ue/Vinf Dstar Theta Cf H``). Points are in panel order:
    starting near the trailing edge on the upper surface, going forward to
    the leading edge, then aft along the lower surface back to the trailing
    edge.

    Attributes:
        operating_point: The operating point this profile was sampled at.
        s: Arclength along the surface.
        x: Chordwise position.
        y: Vertical position.
        ue_vinf: Edge velocity normalised by the freestream velocity.
        delta_star: Displacement thickness.
        theta: Momentum thickness.
        cf: Skin-friction coefficient.
        h: Boundary-layer shape factor (``delta_star / theta``).
    """

    operating_point: OperatingPoint
    s: FloatArray
    x: FloatArray
    y: FloatArray
    ue_vinf: FloatArray
    delta_star: FloatArray
    theta: FloatArray
    cf: FloatArray
    h: FloatArray


@dataclass(slots=True)
class Polar:
    """A sweep of :class:`PolarPoint` instances at a fixed Reynolds/Mach.

    Attributes:
        points: The converged operating-point solutions, in sweep order.
        failed: Operating points that failed to converge.
    """

    points: list[PolarPoint] = field(default_factory=list)
    failed: list[OperatingPoint] = field(default_factory=list)

    def __len__(self) -> int:
        """int: Number of converged points in the polar."""
        return len(self.points)
