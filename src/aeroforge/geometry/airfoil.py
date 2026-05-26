"""The :class:`Airfoil` value object: the central geometry abstraction.

An :class:`Airfoil` stores a 2D contour as paired coordinate arrays in **Selig
order** and exposes derived geometric quantities (thickness, camber, area) plus
I/O and affine transforms. It is deliberately solver-agnostic: generators
produce :class:`Airfoil` instances, and the XFOIL layer consumes them, but
neither is coupled to the other.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aeroforge.core.exceptions import InvalidAirfoilError
from aeroforge.core.types import FloatArray
from aeroforge.geometry import metrics
from aeroforge.geometry.operations import transforms

PathLike = str | Path


class Airfoil:
    """An immutable-by-convention 2D airfoil contour.

    Coordinates are stored in Selig order: starting at the trailing edge,
    running forward over the upper surface to the leading edge, then back over
    the lower surface to the trailing edge. This is the format XFOIL expects.

    Transform methods (:meth:`translated`, :meth:`scaled`, :meth:`rotated`,
    :meth:`normalized`) return **new** instances rather than mutating in place,
    which keeps optimization pipelines free of aliasing bugs.

    Attributes:
        name: Human-readable label (e.g. ``"NACA 2412"``).
    """

    __slots__ = ("_x", "_y", "name")

    def __init__(self, x: FloatArray, y: FloatArray, name: str = "airfoil") -> None:
        """Create an airfoil from coordinate arrays.

        Args:
            x: X coordinates in Selig order.
            y: Y coordinates matching ``x``.
            name: Human-readable label.

        Raises:
            InvalidAirfoilError: If the arrays are mismatched, too short, or
                contain non-finite values.
        """
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        self._validate(x_arr, y_arr)
        self._x = x_arr
        self._y = y_arr
        self.name = name

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate(x: FloatArray, y: FloatArray) -> None:
        """Validate coordinate arrays.

        Args:
            x: Candidate x coordinates.
            y: Candidate y coordinates.

        Raises:
            InvalidAirfoilError: On any malformed input.
        """
        if x.shape != y.shape:
            raise InvalidAirfoilError(
                f"x and y must have the same shape, got {x.shape} and {y.shape}."
            )
        if x.size < 4:
            raise InvalidAirfoilError(f"An airfoil needs >= 4 points, got {x.size}.")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise InvalidAirfoilError("Coordinates contain NaN or infinite values.")

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_arrays(cls, x: FloatArray, y: FloatArray, name: str = "airfoil") -> Airfoil:
        """Alias of the constructor for readability at call sites.

        Args:
            x: X coordinates in Selig order.
            y: Y coordinates matching ``x``.
            name: Human-readable label.

        Returns:
            A new :class:`Airfoil`.
        """
        return cls(x, y, name=name)

    @classmethod
    def from_dat(cls, path: PathLike) -> Airfoil:
        """Load an airfoil from a Selig-format ``.dat`` file.

        The first line is treated as the airfoil name if it is not a numeric
        coordinate pair. All subsequent lines parseable as two floats are read
        as coordinates.

        Args:
            path: Path to the ``.dat`` file.

        Returns:
            A new :class:`Airfoil`.

        Raises:
            InvalidAirfoilError: If the file contains no parseable coordinates
                or appears to use the (unsupported) Lednicer block format.
        """
        path = Path(path)
        name = path.stem
        coords: list[tuple[float, float]] = []
        with path.open("r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    xi, yi = float(parts[0]), float(parts[1])
                except ValueError:
                    if i == 0:  # header / name line
                        name = line.strip()
                    continue
                coords.append((xi, yi))

        if not coords:
            raise InvalidAirfoilError(f"No coordinates found in {path}.")

        arr = np.asarray(coords, dtype=float)
        # Lednicer files start with a 'count' pair like '61. 61.' > 1.
        if arr[0, 0] > 1.5 and arr[0, 1] > 1.5:
            raise InvalidAirfoilError(
                "File appears to use the Lednicer format, which is not yet "
                "supported. Please convert it to Selig order first."
            )
        return cls(arr[:, 0], arr[:, 1], name=name)

    # ------------------------------------------------------------------ #
    # Core data access
    # ------------------------------------------------------------------ #
    @property
    def x(self) -> FloatArray:
        """FloatArray: X coordinates in Selig order (read-only view)."""
        return self._x

    @property
    def y(self) -> FloatArray:
        """FloatArray: Y coordinates in Selig order (read-only view)."""
        return self._y

    @property
    def coordinates(self) -> FloatArray:
        """FloatArray: Coordinates stacked as an ``(N, 2)`` array."""
        return np.column_stack([self._x, self._y])

    @property
    def n_points(self) -> int:
        """int: Number of coordinate points."""
        return int(self._x.size)

    # ------------------------------------------------------------------ #
    # Key landmarks
    # ------------------------------------------------------------------ #
    @property
    def leading_edge(self) -> tuple[float, float]:
        """tuple[float, float]: The ``(x, y)`` of the leading edge (min x)."""
        i = metrics.leading_edge_index(self._x)
        return float(self._x[i]), float(self._y[i])

    @property
    def trailing_edge(self) -> tuple[float, float]:
        """tuple[float, float]: The trailing edge as the mean of the endpoints."""
        return (
            float(0.5 * (self._x[0] + self._x[-1])),
            float(0.5 * (self._y[0] + self._y[-1])),
        )

    @property
    def chord(self) -> float:
        """float: Straight-line distance from leading to trailing edge."""
        le, te = self.leading_edge, self.trailing_edge
        return float(np.hypot(te[0] - le[0], te[1] - le[1]))

    def surfaces(self) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Split the contour into upper and lower surfaces (LE -> TE).

        Returns:
            ``(x_upper, y_upper, x_lower, y_lower)``.
        """
        return metrics.split_surfaces(self._x, self._y)

    # ------------------------------------------------------------------ #
    # Geometric metrics
    # ------------------------------------------------------------------ #
    @property
    def max_thickness(self) -> float:
        """float: Maximum thickness as a fraction of chord."""
        return metrics.max_thickness(self._x, self._y)[0]

    @property
    def max_thickness_location(self) -> float:
        """float: Chordwise location (x/c) of maximum thickness."""
        return metrics.max_thickness(self._x, self._y)[1]

    @property
    def max_camber(self) -> float:
        """float: Maximum (signed) camber as a fraction of chord."""
        return metrics.max_camber(self._x, self._y)[0]

    @property
    def max_camber_location(self) -> float:
        """float: Chordwise location (x/c) of maximum camber."""
        return metrics.max_camber(self._x, self._y)[1]

    @property
    def area(self) -> float:
        """float: Enclosed cross-sectional area (chord squared)."""
        return metrics.enclosed_area(self._x, self._y)

    @property
    def trailing_edge_gap(self) -> float:
        """float: Trailing-edge gap (0 for a closed trailing edge)."""
        return metrics.trailing_edge_gap(self._x, self._y)

    # ------------------------------------------------------------------ #
    # Transforms (return new instances)
    # ------------------------------------------------------------------ #
    def translated(self, dx: float, dy: float) -> Airfoil:
        """Return a copy translated by ``(dx, dy)``.

        Args:
            dx: Shift in x.
            dy: Shift in y.

        Returns:
            A new translated :class:`Airfoil`.
        """
        x, y = transforms.translate(self._x, self._y, dx, dy)
        return Airfoil(x, y, name=self.name)

    def scaled(self, factor: float, origin: tuple[float, float] = (0.0, 0.0)) -> Airfoil:
        """Return a copy uniformly scaled about ``origin``.

        Args:
            factor: Scale factor.
            origin: Fixed point of the scaling.

        Returns:
            A new scaled :class:`Airfoil`.
        """
        x, y = transforms.scale(self._x, self._y, factor, origin=origin)
        return Airfoil(x, y, name=self.name)

    def rotated(self, angle_deg: float, origin: tuple[float, float] = (0.0, 0.0)) -> Airfoil:
        """Return a copy rotated counter-clockwise by ``angle_deg``.

        Args:
            angle_deg: Rotation angle in degrees.
            origin: Center of rotation.

        Returns:
            A new rotated :class:`Airfoil`.
        """
        x, y = transforms.rotate(self._x, self._y, angle_deg, origin=origin)
        return Airfoil(x, y, name=self.name)

    def normalized(self) -> Airfoil:
        """Return a copy with unit chord aligned to the x-axis from (0, 0).

        Translates the leading edge to the origin, rotates so the chord line is
        horizontal, and scales the chord to 1. Useful for comparing airfoils of
        different size or orientation on equal footing.

        Returns:
            A normalized :class:`Airfoil`.
        """
        le = self.leading_edge
        te = self.trailing_edge
        dx, dy = te[0] - le[0], te[1] - le[1]
        chord = float(np.hypot(dx, dy))
        angle = float(np.degrees(np.arctan2(dy, dx)))
        out = self.translated(-le[0], -le[1]).rotated(-angle)
        return out.scaled(1.0 / chord) if chord > 0 else out

    def copy(self) -> Airfoil:
        """Return a deep copy of this airfoil.

        Returns:
            A new :class:`Airfoil` with copied coordinate arrays.
        """
        return Airfoil(self._x.copy(), self._y.copy(), name=self.name)

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def to_dat(self, path: PathLike, *, header: str | None = None) -> Path:
        """Write the airfoil to a Selig-format ``.dat`` file.

        Args:
            path: Destination file path.
            header: Optional first-line label. Defaults to :attr:`name`.

        Returns:
            The path that was written, as a :class:`pathlib.Path`.
        """
        path = Path(path)
        label = header if header is not None else self.name
        lines = [label]
        lines.extend(f"{xi:.6f} {yi:.6f}" for xi, yi in zip(self._x, self._y, strict=False))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    # ------------------------------------------------------------------ #
    # Dunder methods
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        """int: Number of coordinate points."""
        return self.n_points

    def __eq__(self, other: object) -> bool:
        """Compare two airfoils by name and coordinates."""
        if not isinstance(other, Airfoil):
            return NotImplemented
        return (
            self.name == other.name
            and self._x.shape == other._x.shape
            and np.allclose(self._x, other._x)
            and np.allclose(self._y, other._y)
        )

    def __repr__(self) -> str:
        """str: Concise developer-facing representation."""
        return (
            f"Airfoil(name={self.name!r}, n_points={self.n_points}, "
            f"max_thickness={self.max_thickness:.4f}, "
            f"max_camber={self.max_camber:.4f})"
        )
