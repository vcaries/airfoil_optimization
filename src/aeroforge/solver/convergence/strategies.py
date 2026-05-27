"""Concrete convergence-helper strategies.

Each class encodes one well-known XFOIL convergence trick. They share the
:class:`ConvergenceStrategy` interface so they can be composed in any order by
the :class:`ConvergencePipeline`.

All strategies follow the same contract:

* On success: return a converged :class:`PolarPoint`.
* On failure: raise :class:`ConvergenceError` so the pipeline can fall through
  to the next strategy.

Strategies that mutate the underlying solver's parameters do so inside a
``try / finally`` so the original configuration is always restored, which
keeps strategies safe to compose in any order.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import replace
from math import ceil
from typing import TYPE_CHECKING, Any

from aeroforge.core.exceptions import AeroforgeError, ConvergenceError
from aeroforge.core.logging import get_logger
from aeroforge.core.types import OperatingPoint
from aeroforge.solver.convergence.base import ConvergenceStrategy

if TYPE_CHECKING:
    from aeroforge.geometry.airfoil import Airfoil
    from aeroforge.solver.base import AbstractSolver
    from aeroforge.solver.xfoil.results import PolarPoint

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Small helpers shared across strategies
# --------------------------------------------------------------------------- #
@contextmanager
def _temp_attrs(target: Any, **overrides: Any) -> Iterator[None]:
    """Temporarily set attributes on ``target`` and restore them on exit.

    Used by strategies to bump XFOIL parameters (e.g. ``max_iter``) for a
    single retry without permanently changing the solver state.

    Args:
        target: The object whose attributes to override (typically the solver).
        **overrides: Attribute names mapped to their temporary values.

    Yields:
        None. The block runs with the overrides applied; on exit (whether
        normal or exceptional) the original values are restored.
    """
    sentinel = object()
    saved: dict[str, Any] = {}
    try:
        for name, value in overrides.items():
            saved[name] = getattr(target, name, sentinel)
            setattr(target, name, value)
        yield
    finally:
        for name, value in saved.items():
            if value is sentinel:
                # The attribute did not exist before; remove it again.
                with suppress(AttributeError):
                    delattr(target, name)
            else:
                setattr(target, name, value)


def _wrap_convergence_failure(
    strategy: str, point: OperatingPoint, exc: BaseException
) -> ConvergenceError:
    """Build a :class:`ConvergenceError` describing why a strategy failed."""
    return ConvergenceError(
        f"{strategy} failed at alpha={point.alpha:.3f}: {exc}",
        alpha=point.alpha,
    )


# --------------------------------------------------------------------------- #
# Strategy: increase the viscous iteration cap
# --------------------------------------------------------------------------- #
class IncreaseIterationsStrategy(ConvergenceStrategy):
    """Raise XFOIL's ``ITER`` cap and retry.

    Often enough on its own: many "failed to converge in 200 iter" cases just
    needed a few hundred more iterations.

    Args:
        factor: Multiplicative factor applied to the current iteration cap.
        max_iter: Absolute ceiling beyond which the strategy gives up.
    """

    def __init__(self, factor: float = 2.0, max_iter: int = 800) -> None:
        """Store the iteration-bump parameters."""
        if factor <= 1.0:
            raise ValueError("IncreaseIterationsStrategy.factor must be > 1.")
        self.factor = float(factor)
        self.max_iter = int(max_iter)

    def attempt(
        self,
        solver: AbstractSolver,
        airfoil: Airfoil,
        point: OperatingPoint,
        *,
        history: list[PolarPoint],
    ) -> PolarPoint:
        """Retry with a higher iteration cap."""
        current = int(getattr(solver, "max_iter", 200))
        bumped = min(int(current * self.factor), self.max_iter)
        if bumped <= current:
            raise ConvergenceError(
                f"IncreaseIterationsStrategy already at ceiling ({current}).",
                alpha=point.alpha,
            )
        _log.debug(
            "IncreaseIterationsStrategy: %d -> %d at alpha=%.3f",
            current,
            bumped,
            point.alpha,
        )
        try:
            with _temp_attrs(solver, max_iter=bumped):
                return solver.analyze(airfoil, point)
        except ConvergenceError:
            raise
        except AeroforgeError as exc:
            raise _wrap_convergence_failure(self.name, point, exc) from exc


# --------------------------------------------------------------------------- #
# Strategy: alpha continuation from the nearest converged neighbour
# --------------------------------------------------------------------------- #
class AlphaContinuationStrategy(ConvergenceStrategy):
    """Walk alpha in small steps from the nearest converged neighbour.

    XFOIL's boundary-layer solver converges much more reliably when warm-
    started from a nearby converged solution. This strategy steps from the
    closest alpha in ``history`` towards the failing target.

    Args:
        step: Alpha increment (degrees) used during continuation.
        max_steps: Maximum number of intermediate alphas before giving up.
    """

    def __init__(self, step: float = 0.25, max_steps: int = 20) -> None:
        """Store the continuation parameters."""
        if step <= 0.0:
            raise ValueError("AlphaContinuationStrategy.step must be > 0.")
        self.step = float(step)
        self.max_steps = int(max_steps)

    def attempt(
        self,
        solver: AbstractSolver,
        airfoil: Airfoil,
        point: OperatingPoint,
        *,
        history: list[PolarPoint],
    ) -> PolarPoint:
        """Sweep alpha towards the target from the closest converged neighbour."""
        if not history:
            raise ConvergenceError(
                "AlphaContinuationStrategy needs at least one converged "
                "history point to warm-start from.",
                alpha=point.alpha,
            )

        anchor = min(history, key=lambda h: abs(h.operating_point.alpha - point.alpha))
        anchor_alpha = anchor.operating_point.alpha
        direction = 1.0 if point.alpha > anchor_alpha else -1.0
        n_steps = max(1, ceil(abs(point.alpha - anchor_alpha) / self.step))
        if n_steps > self.max_steps:
            raise ConvergenceError(
                f"AlphaContinuationStrategy would need {n_steps} steps "
                f"(> max_steps={self.max_steps}).",
                alpha=point.alpha,
            )

        _log.debug(
            "AlphaContinuationStrategy: %.3f -> %.3f in %d step(s) of %.3f",
            anchor_alpha,
            point.alpha,
            n_steps,
            direction * self.step,
        )

        last_point = anchor
        for k in range(1, n_steps + 1):
            intermediate_alpha = anchor_alpha + direction * self.step * k
            # Don't overshoot the target.
            if (direction > 0 and intermediate_alpha > point.alpha) or (
                direction < 0 and intermediate_alpha < point.alpha
            ):
                intermediate_alpha = point.alpha
            intermediate = replace(point, alpha=intermediate_alpha)
            try:
                last_point = solver.analyze(airfoil, intermediate)
            except AeroforgeError as exc:
                raise _wrap_convergence_failure(self.name, point, exc) from exc
        return last_point


# --------------------------------------------------------------------------- #
# Strategy: nudge alpha by a small epsilon to escape limit cycles
# --------------------------------------------------------------------------- #
class PerturbAlphaStrategy(ConvergenceStrategy):
    """Retry at ``alpha + epsilon`` or ``alpha - epsilon``.

    XFOIL occasionally limit-cycles on a single ITER count near stall; a tiny
    perturbation often breaks the deadlock and gives a result close enough to
    the requested point to be usable.

    Args:
        epsilon: Magnitude of the perturbation, in degrees.
    """

    def __init__(self, epsilon: float = 0.05) -> None:
        """Store the perturbation magnitude."""
        if epsilon <= 0.0:
            raise ValueError("PerturbAlphaStrategy.epsilon must be > 0.")
        self.epsilon = float(epsilon)

    def attempt(
        self,
        solver: AbstractSolver,
        airfoil: Airfoil,
        point: OperatingPoint,
        *,
        history: list[PolarPoint],
    ) -> PolarPoint:
        """Try ``alpha + epsilon`` first, then ``alpha - epsilon``."""
        last_exc: BaseException | None = None
        for sign in (+1.0, -1.0):
            perturbed = replace(point, alpha=point.alpha + sign * self.epsilon)
            try:
                _log.debug("PerturbAlphaStrategy: trying alpha=%.4f", perturbed.alpha)
                return solver.analyze(airfoil, perturbed)
            except AeroforgeError as exc:
                last_exc = exc
                continue
        assert last_exc is not None
        raise _wrap_convergence_failure(self.name, point, last_exc) from last_exc


# --------------------------------------------------------------------------- #
# Strategy: apply XFOIL's auto-repaneling
# --------------------------------------------------------------------------- #
class RepanelStrategy(ConvergenceStrategy):
    """Force XFOIL's ``PANE`` automatic repaneling and retry.

    Useful when the airfoil arrives with awkward panel spacing (e.g. from an
    imported ``.dat`` file) that confuses the BL solver.
    """

    def attempt(
        self,
        solver: AbstractSolver,
        airfoil: Airfoil,
        point: OperatingPoint,
        *,
        history: list[PolarPoint],
    ) -> PolarPoint:
        """Enable repaneling for this single call and retry."""
        try:
            with _temp_attrs(solver, repanel=True):
                return solver.analyze(airfoil, point)
        except ConvergenceError:
            raise
        except AeroforgeError as exc:
            raise _wrap_convergence_failure(self.name, point, exc) from exc


# --------------------------------------------------------------------------- #
# Strategy: inviscid initialisation
# --------------------------------------------------------------------------- #
class InviscidInitStrategy(ConvergenceStrategy):
    """Solve inviscidly first, then retry viscously.

    The inviscid Cp distribution provides a better starting guess for the
    boundary-layer iteration, particularly near stall. We don't keep the
    inviscid result; it's only used to warm up XFOIL's internal state.

    Note:
        XFOIL's internal warm-start would require keeping the binary alive
        across both calls. Since :class:`XfoilRunner` is one-shot today, this
        strategy currently just retries viscously with a bumped iteration
        cap, which captures most of the practical benefit; a true warm-start
        is planned for the next milestone.
    """

    def __init__(self, max_iter_bump: int = 400) -> None:
        """Store the iteration-cap to use for the viscous retry."""
        self.max_iter_bump = int(max_iter_bump)

    def attempt(
        self,
        solver: AbstractSolver,
        airfoil: Airfoil,
        point: OperatingPoint,
        *,
        history: list[PolarPoint],
    ) -> PolarPoint:
        """Approximate the inviscid-warm-start by retrying with more iterations."""
        try:
            with _temp_attrs(solver, max_iter=self.max_iter_bump):
                return solver.analyze(airfoil, point)
        except ConvergenceError:
            raise
        except AeroforgeError as exc:
            raise _wrap_convergence_failure(self.name, point, exc) from exc
