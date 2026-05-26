"""Design-variable definitions used by the optimization problem.

A :class:`DesignVariable` knows its bounds, its identifier, and (optionally) a
human-readable label. The :class:`DesignSpace` aggregates them and exposes the
flat ``(lower, upper)`` vectors expected by pymoo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aeroforge.core.types import FloatArray


@dataclass(frozen=True, slots=True)
class DesignVariable:
    """A continuous design variable with hard bounds.

    Attributes:
        name: Identifier used as a dict key by the evaluator.
        lower: Lower bound (inclusive).
        upper: Upper bound (inclusive).
        label: Optional pretty label for plots and reports.
    """

    name: str
    lower: float
    upper: float
    label: str | None = None

    def __post_init__(self) -> None:
        """Validate bounds."""
        if not self.upper > self.lower:
            raise ValueError(
                f"DesignVariable {self.name!r}: upper ({self.upper}) must be > "
                f"lower ({self.lower})."
            )


@dataclass(slots=True)
class DesignSpace:
    """An ordered collection of design variables.

    Attributes:
        variables: Ordered list of :class:`DesignVariable` instances.
    """

    variables: list[DesignVariable] = field(default_factory=list)

    @property
    def n_var(self) -> int:
        """int: Number of design variables."""
        return len(self.variables)

    @property
    def bounds(self) -> tuple[FloatArray, FloatArray]:
        """tuple[FloatArray, FloatArray]: Parallel ``(lower, upper)`` arrays."""
        lower = np.asarray([v.lower for v in self.variables], dtype=float)
        upper = np.asarray([v.upper for v in self.variables], dtype=float)
        return lower, upper

    @property
    def names(self) -> list[str]:
        """list[str]: Variable names in order."""
        return [v.name for v in self.variables]

    def to_mapping(self, x: FloatArray) -> dict[str, float]:
        """Convert a flat genome vector to a ``{name: value}`` mapping.

        Args:
            x: A length-``n_var`` array in the order of :attr:`variables`.

        Returns:
            A dictionary keyed by variable name.

        Raises:
            ValueError: If ``len(x) != n_var``.
        """
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.n_var:
            raise ValueError(f"Expected {self.n_var} values, got {x.size}.")
        return dict(zip(self.names, x.tolist(), strict=True))
