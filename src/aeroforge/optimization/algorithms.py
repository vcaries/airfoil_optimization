"""Algorithm factories.

Wrappers around pymoo's algorithm constructors so user code does not depend on
pymoo's internal module paths and so we can tweak default hyper-parameters in
one place. Each factory returns a configured, ready-to-use pymoo algorithm.

Requires the ``optim`` extra.
"""

from __future__ import annotations

from typing import Any


def nsga2(pop_size: int = 40, **kwargs: Any) -> Any:
    """Return a configured pymoo NSGA-II algorithm.

    Args:
        pop_size: Population size.
        **kwargs: Forwarded to :class:`pymoo.algorithms.moo.nsga2.NSGA2`.

    Returns:
        A pymoo algorithm instance.
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2  # type: ignore[import-not-found]

    return NSGA2(pop_size=pop_size, **kwargs)


def ga(pop_size: int = 40, **kwargs: Any) -> Any:
    """Return a configured pymoo single-objective GA.

    Args:
        pop_size: Population size.
        **kwargs: Forwarded to :class:`pymoo.algorithms.soo.nonconvex.ga.GA`.

    Returns:
        A pymoo algorithm instance.
    """
    from pymoo.algorithms.soo.nonconvex.ga import GA  # type: ignore[import-not-found]

    return GA(pop_size=pop_size, **kwargs)


def nsga3(ref_dirs: Any, pop_size: int = 92, **kwargs: Any) -> Any:
    """Return a configured pymoo NSGA-III algorithm (3+ objectives).

    Args:
        ref_dirs: Reference directions (from :mod:`pymoo.util.ref_dirs`).
        pop_size: Population size.
        **kwargs: Forwarded to :class:`pymoo.algorithms.moo.nsga3.NSGA3`.

    Returns:
        A pymoo algorithm instance.
    """
    from pymoo.algorithms.moo.nsga3 import NSGA3  # type: ignore[import-not-found]

    return NSGA3(ref_dirs=ref_dirs, pop_size=pop_size, **kwargs)
