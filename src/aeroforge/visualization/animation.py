"""Animation pipeline for portfolio-grade GIFs and MP4s.

The animations are the visible payoff of the project. The pipeline turns an
optimization :class:`HistoryCallback` into:

* the best-so-far airfoil geometry evolving generation by generation,
* the Pareto front advancing through the objective space.

Output format is auto-detected from the file extension: ``.gif`` uses
:mod:`imageio`, ``.mp4`` uses :mod:`imageio` with the ``ffmpeg`` plugin
(provided by :mod:`imageio_ffmpeg`).

Memory is bounded by rendering each frame into a fresh figure that is closed
immediately after the frame is appended, so a 200-generation animation stays
well below 100 MB regardless of figure complexity.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from aeroforge.core.exceptions import AeroforgeError
from aeroforge.core.logging import get_logger
from aeroforge.visualization.pareto import non_dominated_mask
from aeroforge.visualization.plots import plot_geometry
from aeroforge.visualization.style import (
    PORTFOLIO_PALETTE,
    use_portfolio_style,
)

if TYPE_CHECKING:
    from aeroforge.geometry.airfoil import Airfoil
    from aeroforge.optimization.callbacks import HistoryCallback

PathLike = str | Path
_log = get_logger(__name__)

# Sentinel function signature: genome -> Airfoil.
GenomeToAirfoil = Callable[[np.ndarray], "Airfoil"]


# --------------------------------------------------------------------------- #
# Internal frame writer
# --------------------------------------------------------------------------- #
def _write_frames(
    frames: list[np.ndarray],
    output: PathLike,
    *,
    fps: int,
) -> Path:
    """Stitch a list of RGB frames into a GIF or MP4 at ``output``."""
    import imageio.v2 as imageio  # type: ignore[import-not-found]

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(out_path, frames, duration=1.0 / fps, loop=0)  # type: ignore[arg-type]
    elif suffix in {".mp4", ".m4v"}:
        with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8) as writer:
            for frame in frames:
                writer.append_data(frame)  # type: ignore[attr-defined]
    else:
        raise AeroforgeError(f"Unsupported animation extension {suffix!r}; use .gif or .mp4.")
    _log.info("Wrote %d frames at %d fps to %s", len(frames), fps, out_path)
    return out_path


def _figure_to_rgb(fig: Any) -> np.ndarray:
    """Rasterise a Matplotlib figure to an RGB ndarray, then close it."""
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    import imageio.v2 as imageio  # type: ignore[import-not-found]

    return np.asarray(imageio.imread(buf))


# --------------------------------------------------------------------------- #
# Geometry evolution
# --------------------------------------------------------------------------- #
def animate_geometry_evolution(
    history: HistoryCallback,
    genome_to_airfoil: GenomeToAirfoil,
    output: PathLike,
    *,
    fps: int = 12,
    objective_index: int = 0,
    show_baseline: bool = True,
) -> Path:
    """Animate the best-so-far airfoil shape across generations.

    Args:
        history: The :class:`HistoryCallback` collected during the run.
        genome_to_airfoil: Callable that decodes a genome vector into an
            :class:`Airfoil`. Typically
            :meth:`AirfoilEvaluator.genome_to_airfoil`.
        output: Output file path (``.gif`` or ``.mp4``).
        fps: Frames per second.
        objective_index: Which objective column to minimise when picking the
            best genome in each generation (only relevant for multi-objective
            studies).
        show_baseline: When ``True``, overlay the generation-0 best airfoil
            in faded outline on every frame so the viewer can gauge how much
            the geometry has changed.

    Returns:
        The path that was written.

    Raises:
        AeroforgeError: If ``history`` is empty or the extension is unknown.
    """
    if not history.snapshots:
        raise AeroforgeError("HistoryCallback is empty; nothing to animate.")

    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    use_portfolio_style()

    # Pre-compute the best airfoil per generation so the y-axis can be
    # locked across frames (a jumping y-axis is the #1 reader-confusing
    # animation antipattern).
    best_airfoils: list[Airfoil] = []
    for snap in history.snapshots:
        idx = int(np.argmin(np.asarray(snap.f, dtype=float)[:, objective_index]))
        best_airfoils.append(genome_to_airfoil(np.asarray(snap.x[idx], dtype=float)))

    ymin = min(float(np.min(af.y)) for af in best_airfoils) - 0.02
    ymax = max(float(np.max(af.y)) for af in best_airfoils) + 0.02
    baseline = best_airfoils[0] if show_baseline else None

    frames: list[np.ndarray] = []
    for gen, af in enumerate(best_airfoils):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        if baseline is not None and gen > 0:
            ax.plot(
                baseline.x,
                baseline.y,
                color="0.7",
                lw=1.0,
                ls="--",
                label=f"gen 0 ({baseline.name})",
            )
        plot_geometry(af, ax=ax, label=f"gen {gen} (best)")
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"Best geometry — generation {gen}")
        ax.legend(loc="upper right", fontsize="small")
        frames.append(_figure_to_rgb(fig))

    return _write_frames(frames, output, fps=fps)


# --------------------------------------------------------------------------- #
# Pareto-front evolution
# --------------------------------------------------------------------------- #
def animate_pareto_evolution(
    history: HistoryCallback,
    output: PathLike,
    *,
    fps: int = 12,
    xlabel: str = r"$f_1$",
    ylabel: str = r"$f_2$",
) -> Path:
    """Animate the Pareto front as it advances across generations.

    Args:
        history: The :class:`HistoryCallback` collected during the run.
        output: Output file path (``.gif`` or ``.mp4``).
        fps: Frames per second.
        xlabel: Axis label for the first objective.
        ylabel: Axis label for the second objective.

    Returns:
        The path that was written.

    Raises:
        AeroforgeError: If the history is empty or the snapshots have fewer
            than two objectives.
    """
    if not history.snapshots:
        raise AeroforgeError("HistoryCallback is empty; nothing to animate.")
    if history.snapshots[0].f.shape[1] < 2:
        raise AeroforgeError(
            "animate_pareto_evolution needs >= 2 objectives, "
            f"got shape={history.snapshots[0].f.shape}."
        )

    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    use_portfolio_style()

    # Lock axis limits across frames to avoid distracting rescaling.
    all_f = np.vstack([np.asarray(s.f, dtype=float) for s in history.snapshots])
    x_lo, x_hi = float(np.min(all_f[:, 0])), float(np.max(all_f[:, 0]))
    y_lo, y_hi = float(np.min(all_f[:, 1])), float(np.max(all_f[:, 1]))
    x_pad = 0.05 * (x_hi - x_lo + 1e-12)
    y_pad = 0.05 * (y_hi - y_lo + 1e-12)

    frames: list[np.ndarray] = []
    for snap in history.snapshots:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        f = np.asarray(snap.f, dtype=float)
        mask = non_dominated_mask(f)
        ax.scatter(
            f[~mask, 0],
            f[~mask, 1],
            s=18,
            color="0.65",
            alpha=0.7,
            label="dominated",
        )
        ax.scatter(
            f[mask, 0],
            f[mask, 1],
            s=42,
            color=PORTFOLIO_PALETTE[3],
            edgecolors="white",
            lw=0.8,
            label="non-dominated",
        )
        ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Pareto front — generation {snap.generation}")
        ax.legend(loc="best", fontsize="small")
        frames.append(_figure_to_rgb(fig))

    return _write_frames(frames, output, fps=fps)
