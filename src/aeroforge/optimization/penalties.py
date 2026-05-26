"""Penalty functions for soft-constraint handling.

Pymoo natively supports hard constraints (the ``<= 0`` convention) but soft
penalties are often more numerically stable for evolutionary algorithms,
particularly when constraint evaluation involves an expensive solver call.
"""

from __future__ import annotations

from collections.abc import Callable

PenaltyFn = Callable[[float], float]


def quadratic_penalty(weight: float = 1.0) -> PenaltyFn:
    """Return a quadratic penalty ``weight * max(g, 0)**2``.

    Args:
        weight: Multiplicative weight applied to the penalty.

    Returns:
        A callable ``g -> penalty`` that is zero in the feasible region.
    """

    def _penalty(g: float) -> float:
        return weight * max(g, 0.0) ** 2

    return _penalty


def linear_penalty(weight: float = 1.0) -> PenaltyFn:
    """Return a linear penalty ``weight * max(g, 0)``.

    Args:
        weight: Multiplicative weight applied to the penalty.

    Returns:
        A callable ``g -> penalty`` that is zero in the feasible region.
    """

    def _penalty(g: float) -> float:
        return weight * max(g, 0.0)

    return _penalty


def exponential_penalty(weight: float = 1.0, scale: float = 10.0) -> PenaltyFn:
    """Return an exponential penalty ``weight * (exp(scale * max(g, 0)) - 1)``.

    Useful when even small infeasibilities should be aggressively discouraged.

    Args:
        weight: Multiplicative weight applied to the penalty.
        scale: Steepness of the exponential.

    Returns:
        A callable ``g -> penalty`` that is zero in the feasible region.
    """
    import math

    def _penalty(g: float) -> float:
        return weight * (math.expm1(scale * max(g, 0.0)))

    return _penalty
