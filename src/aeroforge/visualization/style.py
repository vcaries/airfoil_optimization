"""Matplotlib style configuration for portfolio-grade plots.

Centralising the style here gives every figure a consistent identity (fonts,
colours, line widths, grid) without scattering ``rcParams`` calls throughout
the codebase.

Requires the ``viz`` extra (``pip install aeroforge[viz]``).
"""

from __future__ import annotations

# Curated qualitative palette (colour-blind safe; works on light & dark themes).
PORTFOLIO_PALETTE: tuple[str, ...] = (
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
)


def use_portfolio_style() -> None:
    """Apply the aeroforge portfolio plotting style globally.

    Idempotent and safe to call from notebooks, scripts, and tests.
    """
    import matplotlib as mpl  # type: ignore[import-not-found]
    from cycler import cycler  # type: ignore[import-not-found]

    mpl.rcParams.update(
        {
            "figure.figsize": (7.0, 4.5),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "lines.linewidth": 1.75,
            "axes.prop_cycle": cycler(color=list(PORTFOLIO_PALETTE)),
        }
    )
