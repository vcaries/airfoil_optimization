"""Animation pipeline for portfolio-grade GIFs and MP4s.

The animations are the visible payoff of the project. The pipeline turns an
optimization :class:`HistoryCallback` into:

* the optimal geometry evolving generation by generation,
* the Pareto-front advancing through the objective space,
* per-generation aerodynamic-metric trajectories,
* (optionally) the Cp distribution of the current best design.

Output is via :mod:`imageio` (GIF) and :mod:`imageio_ffmpeg` (MP4), both
optional dependencies declared in the ``viz`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aeroforge.optimization.callbacks import HistoryCallback

PathLike = str | Path


def animate_geometry_evolution(
    history: HistoryCallback,
    output: PathLike,
    *,
    fps: int = 12,
) -> Path:
    """Animate the best-so-far airfoil shape across generations.

    Args:
        history: The :class:`HistoryCallback` collected during the run.
        output: Output file path (``.gif`` or ``.mp4``).
        fps: Frames per second.

    Returns:
        The path that was written.

    Raises:
        NotImplementedError: Implementation planned for milestone M5.
    """
    raise NotImplementedError("animate_geometry_evolution (planned, M5).")


def animate_pareto_evolution(
    history: HistoryCallback,
    output: PathLike,
    *,
    fps: int = 12,
) -> Path:
    """Animate the Pareto front as it advances across generations.

    Args:
        history: The :class:`HistoryCallback` collected during the run.
        output: Output file path (``.gif`` or ``.mp4``).
        fps: Frames per second.

    Returns:
        The path that was written.

    Raises:
        NotImplementedError: Implementation planned for milestone M5.
    """
    raise NotImplementedError("animate_pareto_evolution (planned, M5).")
