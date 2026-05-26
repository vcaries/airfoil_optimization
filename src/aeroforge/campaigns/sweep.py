"""Parametric sweep definitions.

A :class:`Sweep` is a Cartesian product over named parameter ranges and a
factory that turns each tuple into a concrete (airfoil, operating point) pair
to evaluate. The campaign runner consumes any iterable of these.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any

from aeroforge.core.types import OperatingPoint

if TYPE_CHECKING:
    from aeroforge.geometry.airfoil import Airfoil


@dataclass(slots=True)
class Sweep:
    """A Cartesian product over named parameter ranges.

    Args:
        parameters: Mapping from parameter name to the list of values to sweep.
        factory: Callable that turns a parameter mapping into an
            ``(Airfoil, OperatingPoint)`` pair. Lets callers control how the
            parameters map onto airfoil / operating-point construction.

    Example:
        >>> sweep = Sweep(
        ...     parameters={"alpha": [0.0, 2.0, 4.0], "re": [3e5, 1e6]},
        ...     factory=lambda p: (airfoil, OperatingPoint(p["alpha"], p["re"])),
        ... )
        >>> for airfoil, point in sweep:
        ...     ...  # doctest: +SKIP
    """

    parameters: dict[str, list[Any]] = field(default_factory=dict)
    factory: Callable[[dict[str, Any]], tuple[Airfoil, OperatingPoint]] | None = None

    def __iter__(self) -> Iterator[tuple[Airfoil, OperatingPoint]]:
        """Yield ``(airfoil, operating_point)`` pairs across the full product.

        Raises:
            ValueError: If :attr:`factory` is not set.
        """
        if self.factory is None:
            raise ValueError("Sweep.factory must be set before iteration.")
        names = list(self.parameters.keys())
        for values in product(*(self.parameters[n] for n in names)):
            yield self.factory(dict(zip(names, values, strict=True)))

    def __len__(self) -> int:
        """int: Total number of points in the sweep."""
        size = 1
        for values in self.parameters.values():
            size *= len(values)
        return size
