"""Visualization layer: static plots and animation pipeline.

Requires the ``viz`` extra (``pip install aeroforge[viz]``). Not imported by
the top-level :mod:`aeroforge` package, so the core library remains usable in
headless / no-matplotlib environments.
"""

from aeroforge.visualization.animation import (
    animate_geometry_evolution,
    animate_pareto_evolution,
)
from aeroforge.visualization.pareto import plot_pareto_front
from aeroforge.visualization.plots import (
    plot_convergence_history,
    plot_cp,
    plot_geometry,
    plot_polar,
)
from aeroforge.visualization.style import PORTFOLIO_PALETTE, use_portfolio_style

__all__ = [
    "use_portfolio_style",
    "PORTFOLIO_PALETTE",
    "plot_geometry",
    "plot_polar",
    "plot_cp",
    "plot_convergence_history",
    "plot_pareto_front",
    "animate_geometry_evolution",
    "animate_pareto_evolution",
]
