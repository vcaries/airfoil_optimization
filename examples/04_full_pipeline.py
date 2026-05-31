"""Multi-physics airfoil portfolio demo — mono-point AND multi-point workflow.

This is the v3 showcase script. It runs N + 1 optimisations and produces a
gallery of consistent, publication-quality visualisations:

  STAGE 1 — Mono-point (single objective pair at one mission point):
            one full optimisation per mission point, each producing the
            complete asset set in its own subdirectory.

  STAGE 2 — Multi-point (aggregated across all mission points):
            one optimisation whose objectives are the lift-weighted total
            CL and the cruise CD, producing the same complete asset set.

  STAGE 3 — Cross-comparison (auto-generated):
            renders comparative figures so the reader can immediately see
            what the multi-point optimum trades against the four
            specialised mono-point optima.

NUMBER OF MISSION POINTS — fully scalable. Edit :data:`MISSION` (any N >= 1)
and every loop, animation, layout, and comparison figure adapts.

ARCHITECTURE — why the evaluator has a single ``operating_point``:
The :class:`AirfoilEvaluator` is single-point by design. Both the
mono-point and the multi-point solvers wrap that contract: they implement
``analyze`` themselves and IGNORE the incoming ``operating_point`` (a
placeholder), then internally evaluate at one or more mission legs and
return an aggregated :class:`PolarPoint`.

The aggregated CL fed to :class:`MaximizeLift` is::

    cl  =  Σ_i  lift_weight_i · CL_i(MissionPoint_i)

For mono-point optimisations, the sum collapses to a single term.

VISUAL STYLE
------------
Computer Modern serif theme (LaTeX-like). Unicode minus disabled to avoid
the "square with X" artefact on systems lacking the U+2212 glyph. Single
shared palette. Every figure uses identical terminology, fonts, marker
sizes, and legend categories.

OUTPUT LAYOUT
-------------
``--out`` directory (default ``docs/assets``) contains::

    docs/assets/
    ├── stage1_takeoff/          PNGs + GIFs for the take-off-only run
    ├── stage1_cruise/           ditto for cruise
    ├── stage1_landing/          ditto for landing
    ├── stage2_multipoint/       PNGs + GIFs for the multi-point run
    └── comparison/              cross-comparison PNGs of all 4 champions

Run it::

    python examples/04_full_pipeline.py                      # all stages, real XFOIL
    python examples/04_full_pipeline.py --synthetic          # closed-form fallback
    python examples/04_full_pipeline.py --pop 40 --n-gen 20  # larger run

By default the script drives the real XFOIL binary (looked up on ``PATH``
under the name configured by ``--xfoil-binary``). Pass ``--synthetic`` to
force the closed-form aerodynamic model — useful on CI, or on machines
where XFOIL is not installed.
"""

from __future__ import annotations

import argparse
import io
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — headless

import imageio.v2 as imageio  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from aeroforge.core.types import OperatingPoint
from aeroforge.geometry import Airfoil, NACA4Generator
from aeroforge.optimization import (
    AirfoilEvaluator,
    DesignSpace,
    DesignVariable,
    GeometricConstraint,
    MaximizeLift,
    MinimizeDrag,
    MinThicknessConstraint,
    OptimizationStudy,
    PhysicalConstraint,
)
from aeroforge.optimization.algorithms import nsga2
from aeroforge.solver.base import AbstractSolver
from aeroforge.solver.xfoil.results import PolarPoint
from aeroforge.visualization.pareto import non_dominated_mask


# ========================================================================== #
# 1. CONFIGURATION — mission profile + variable labels (single source of truth)
# ========================================================================== #
@dataclass(frozen=True, slots=True)
class MissionPoint:
    """One leg of the mission profile."""

    name: str  # short identifier: "takeoff", "cruise", ...
    pretty: str  # display label: "Take-off"
    alpha: float  # AoA (deg)
    reynolds: float  # chord-based Reynolds number
    mach: float  # free-stream Mach
    lift_weight: float  # weight in multi-point aggregation


# Editable: extend, shorten, or reorder freely — every figure adapts.
MISSION: tuple[MissionPoint, ...] = (
    MissionPoint("takeoff", "Take-off", alpha=8.0, reynolds=5.0e5, mach=0.15, lift_weight=1.0),
    MissionPoint("cruise", "Cruise", alpha=2.0, reynolds=2.0e6, mach=0.60, lift_weight=1.5),
    MissionPoint("landing", "Landing", alpha=10.0, reynolds=4.0e5, mach=0.15, lift_weight=1.2),
)


# Variable / metric labelling — used EVERYWHERE for consistency.
VAR = {
    "m": (r"$m$", "Maximum Camber"),
    "p": (r"$p$", "Camber Position"),
    "t": (r"$t$", "Thickness"),
    "cl": (r"$C_l$", "Lift Coefficient"),
    "cd": (r"$C_d$", "Drag Coefficient"),
    "cm": (r"$C_m$", "Pitching Moment"),
    "ld": (r"$L/D$", "Lift-to-Drag Ratio"),
    "alpha": (r"$\alpha$", "Angle of Attack"),
    "cl_tot": (r"$C_{L,\mathrm{tot}}$", "Mission-Weighted Lift"),
    "cd_cr": (r"$C_d$ cruise", "Cruise Drag"),
}


def axis_label(key: str, with_units: str = "") -> str:
    """Return ``'symbol  (Description) [units]'`` for axis labels."""
    sym, desc = VAR[key]
    unit = f"  [{with_units}]" if with_units else ""
    return rf"{sym}  ({desc}){unit}"


# ========================================================================== #
# 2. VISUAL THEME — Computer Modern serif, unicode-minus disabled, single palette
# ========================================================================== #
PALETTE = {
    # Categorical colours used everywhere
    "infeasible": "#c0392b",  # firm red
    "feasible": "#445566",  # cool slate gray
    "pareto": "#e6892a",  # warm amber
    "recommended": "#1f4e8c",  # deep navy — the selected design
    # Faded variants for cumulative scatter
    "ghost_inf": "#f1c7c0",
    "ghost_feas": "#d4d9e0",
    "ghost_par": "#fae0bf",
    "geom_history": "#bdc3cf",
    # Per-mission palette (used wherever colours encode mission identity)
    "mission": ("#1f4e8c", "#a23737", "#196b3a", "#7a3f8c", "#0f8a8a", "#8c5a1f"),
}


def apply_theme() -> None:
    """Apply a serif + Computer Modern math theme; disable unicode minus.

    The ``axes.unicode_minus = False`` is critical: many serif fonts lack
    the U+2212 glyph, which is what was rendering as "square with X" on the
    user's machine. Setting this to False makes matplotlib emit the plain
    ASCII hyphen instead.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "CMU Serif",
                "Computer Modern Roman",
                "DejaVu Serif",
                "Times New Roman",
            ],
            "mathtext.fontset": "cm",  # Computer Modern math (no LaTeX install needed)
            "mathtext.default": "regular",
            "axes.unicode_minus": False,  # ← fixes the "square-with-X" minus rendering
            "axes.titlesize": 12.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11.0,
            "axes.labelpad": 4.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "legend.frameon": True,
            "legend.framealpha": 0.85,
            "legend.edgecolor": "0.7",
            "figure.dpi": 110,
            "savefig.dpi": 170,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "grid.color": "0.6",
            "lines.linewidth": 1.7,
            "axes.facecolor": "#fbfbfc",
            "figure.facecolor": "white",
        }
    )


# ========================================================================== #
# 3. SOLVERS — generic single-point + aggregator over any list[MissionPoint]
# ========================================================================== #
@dataclass
class SyntheticPhysicalModel:
    """Closed-form CL/CD/CM model used by the synthetic solvers."""

    def analyze_single(
        self, airfoil: Airfoil, alpha: float, re: float, mach: float
    ) -> tuple[float, float, float]:
        t = float(airfoil.max_thickness)
        c = float(airfoil.max_camber)
        cl_alpha = 0.110 * (1.0 - 0.6 * t)
        cl0 = 6.28 * c
        cl_lin = cl_alpha * alpha + cl0
        alpha_stall = 10.0 + 50.0 * t
        if alpha > alpha_stall:
            cl = cl_lin * float(np.exp(-(((alpha - alpha_stall) / 5.0) ** 2)))
        else:
            cl = cl_lin
        cd = 0.0055 + 0.02 * (t - 0.12) ** 2 + 0.012 * (c - 0.03) ** 2
        cd += 0.0008 * max(alpha - 4.0, 0.0) ** 2
        cd *= 0.5 + 0.5 * (5.0e5 / max(re, 1.0)) ** 0.5
        if mach < 0.6:
            cd /= max(float(np.sqrt(1.0 - mach * mach)), 1.0e-3)
        else:
            cd /= float(np.sqrt(1.0 - 0.36))
            cd *= 1.0 + 12.0 * (mach - 0.6) ** 2
        cm = -0.05 - 0.6 * c
        return float(cl), float(cd), float(cm)


@dataclass
class AggregatedSyntheticSolver(AbstractSolver):
    """Generic aggregator over an arbitrary list of mission points.

    For mono-point optimisations pass a list with one element; for the
    multi-point optimisation pass the full :data:`MISSION` tuple. The
    aggregated PolarPoint carries::

        cl  =  Σ lift_weight_i · CL_i
        cd  =  CD at the chosen "primary" mission point (default: first)
        cm  =  CM at the primary mission point

    The primary point is used because CD/CM only really make sense at one
    operating condition; for mono-point optimisations the primary is the
    only point. For multi-point, "cruise" is the natural primary.
    """

    mission: tuple[MissionPoint, ...]
    primary_name: str
    model: SyntheticPhysicalModel = field(default_factory=SyntheticPhysicalModel)

    def analyze_single(
        self, airfoil: Airfoil, alpha: float, re: float, mach: float
    ) -> tuple[float, float, float]:
        return self.model.analyze_single(airfoil, alpha, re, mach)

    def analyze(self, airfoil: Airfoil, point: OperatingPoint) -> PolarPoint:
        per_point = {
            mp.name: self.model.analyze_single(airfoil, mp.alpha, mp.reynolds, mp.mach)
            for mp in self.mission
        }
        cl_total = sum(mp.lift_weight * per_point[mp.name][0] for mp in self.mission)
        primary = per_point[self.primary_name]
        return PolarPoint(
            operating_point=point,
            cl=float(cl_total),
            cd=float(primary[1]),
            cdp=float(primary[1] / 8.0),
            cm=float(primary[2]),
            x_trans_upper=0.4 - 0.5 * float(airfoil.max_camber),
            x_trans_lower=0.6 - 0.5 * float(airfoil.max_camber),
            converged=True,
        )


@dataclass
class AggregatedXfoilSolver(AbstractSolver):
    """Real-XFOIL aggregator with full parameter exposure."""

    mission: tuple[MissionPoint, ...]
    primary_name: str
    binary: str = "xfoil"
    max_iter: int = 200
    n_crit: float = 9.0
    repanel: bool = True
    timeout_s: float = 30.0
    x_trip_upper: float | None = None
    x_trip_lower: float | None = None

    def _runner(self) -> Any:
        from aeroforge.solver import XfoilRunner

        return XfoilRunner(
            self.binary,
            max_iter=self.max_iter,
            n_crit=self.n_crit,
            repanel=self.repanel,
            timeout_s=self.timeout_s,
        )

    def analyze_single(
        self, airfoil: Airfoil, alpha: float, re: float, mach: float
    ) -> tuple[float, float, float]:
        op = OperatingPoint(
            alpha=alpha,
            reynolds=re,
            mach=mach,
            n_crit=self.n_crit,
            x_trip_upper=self.x_trip_upper,
            x_trip_lower=self.x_trip_lower,
        )
        pt = self._runner().analyze(airfoil, op)
        return float(pt.cl), float(pt.cd), float(pt.cm)

    def analyze(self, airfoil: Airfoil, point: OperatingPoint) -> PolarPoint:
        per_point: dict[str, tuple[float, float, float] | None] = {}
        for mp in self.mission:
            try:
                per_point[mp.name] = self.analyze_single(airfoil, mp.alpha, mp.reynolds, mp.mach)
            except Exception:  # noqa: BLE001
                per_point[mp.name] = None
        primary = per_point[self.primary_name]
        if primary is None:
            raise RuntimeError(f"Primary mission point '{self.primary_name}' failed.")
        cl_total = sum(
            mp.lift_weight * (per_point[mp.name][0] if per_point[mp.name] else 0.0)
            for mp in self.mission
        )
        return PolarPoint(
            operating_point=point,
            cl=float(cl_total),
            cd=float(primary[1]),
            cdp=float(primary[1] / 8.0),
            cm=float(primary[2]),
            x_trans_upper=0.3,
            x_trans_lower=0.6,
            converged=True,
        )


def per_point_analysis(
    solver: AbstractSolver, airfoil: Airfoil, mission: tuple[MissionPoint, ...]
) -> dict[str, tuple[float, float, float]]:
    """Re-evaluate ``airfoil`` at every mission point individually."""
    out: dict[str, tuple[float, float, float]] = {}
    for mp in mission:
        try:
            out[mp.name] = solver.analyze_single(  # type: ignore[attr-defined]
                airfoil, mp.alpha, mp.reynolds, mp.mach
            )
        except Exception:  # noqa: BLE001
            out[mp.name] = (float("nan"), float("nan"), float("nan"))
    return out


# ========================================================================== #
# 4. CONSTRAINTS — mechanical/manufacturing (geometry) + engineering (physics)
# ========================================================================== #
# Geometric — invariant across missions: manufacturability and structure.
@dataclass(slots=True)
class MinEnclosedAreaConstraint(GeometricConstraint):
    """Spar-box volume proxy: enclosed cross-section >= ``area_min``."""

    area_min: float

    def evaluate(self, airfoil: Airfoil) -> float:
        return float(self.area_min - airfoil.area)


@dataclass(slots=True)
class MinThicknessAtConstraint(GeometricConstraint):
    """Spar thickness: local t at ``x_target`` >= ``t_min``."""

    x_target: float
    t_min: float

    def evaluate(self, airfoil: Airfoil) -> float:
        xu, yu, xl, yl = airfoil.surfaces()
        yu_t = float(np.interp(self.x_target, xu, yu))
        yl_t = float(np.interp(self.x_target, xl, yl))
        return float(self.t_min - (yu_t - yl_t))


@dataclass(slots=True)
class MaxTEGapConstraint(GeometricConstraint):
    """Manufacturing: trailing-edge gap <= ``gap_max``."""

    gap_max: float

    def evaluate(self, airfoil: Airfoil) -> float:
        return float(airfoil.trailing_edge_gap - self.gap_max)


@dataclass(slots=True)
class MaxAbsCamberConstraint(GeometricConstraint):
    """Structural: abs(max camber) <= ``c_max``."""

    c_max: float

    def evaluate(self, airfoil: Airfoil) -> float:
        return float(abs(airfoil.max_camber) - self.c_max)


# Physical — engineering envelope evaluated on the *aggregated* PolarPoint
# returned by AggregatedSyntheticSolver / AggregatedXfoilSolver. Each
# mission imposes a different combination, which is what makes the
# stage-1 optima genuinely different airfoils.
@dataclass(slots=True)
class MinLiftConstraint(PhysicalConstraint):
    """Aircraft must lift the design weight at the operating point.

    Feasible iff ``result.cl >= cl_min``. The aggregated CL already
    embeds the mission lift weights, so callers must scale the target
    by the same weight (or sum of weights, for multi-point).
    """

    cl_min: float

    def evaluate(self, result: PolarPoint) -> float:
        return float(self.cl_min - result.cl)


@dataclass(slots=True)
class MaxDragConstraint(PhysicalConstraint):
    """Efficiency floor: drag at the primary operating point <= ``cd_max``."""

    cd_max: float

    def evaluate(self, result: PolarPoint) -> float:
        return float(result.cd - self.cd_max)


@dataclass(slots=True)
class MaxAbsPitchingMomentConstraint(PhysicalConstraint):
    """Trim-authority budget: ``|C_m|`` at the primary point <= ``cm_abs_max``."""

    cm_abs_max: float

    def evaluate(self, result: PolarPoint) -> float:
        return float(abs(result.cm) - self.cm_abs_max)


# Per-mission engineering envelopes. Each entry encodes "what is the
# *physical* requirement on this leg" — the take-off entry is high-lift
# permissive on drag; the cruise entry is the opposite; landing pushes
# CL even harder. These are the numbers that drive the differentiation
# of the three stage-1 optima.
ENGINEERING_LIMITS: dict[str, dict[str, float]] = {
    "takeoff": {"cl_min": 1.00, "cd_max": 0.045, "cm_abs_max": 0.15},
    "cruise": {"cl_min": 0.45, "cd_max": 0.018, "cm_abs_max": 0.08},
    "landing": {"cl_min": 1.40, "cd_max": 0.080, "cm_abs_max": 0.20},
}
# Fallback used if a mission point has a custom name not in the table above.
_DEFAULT_LIMITS = {"cl_min": 0.50, "cd_max": 0.030, "cm_abs_max": 0.15}


def standard_geometric_constraints() -> list[GeometricConstraint]:
    """The mechanical/manufacturing envelope — invariant across missions."""
    return [
        MinThicknessConstraint(t_min=0.05),
        MinThicknessAtConstraint(x_target=0.10, t_min=0.06),
        MinEnclosedAreaConstraint(area_min=0.040),
        MaxTEGapConstraint(gap_max=0.005),
        MaxAbsCamberConstraint(c_max=0.08),
    ]


def engineering_physical_constraints(
    mission: tuple[MissionPoint, ...], primary_name: str
) -> list[PhysicalConstraint]:
    """Build the mission-specific aerodynamic envelope.

    Three constraints per run — CL floor (must lift), CD ceiling
    (efficiency floor), and ``|C_m|`` ceiling (trim authority budget).
    The thresholds come from :data:`ENGINEERING_LIMITS`, looked up via
    the *primary* mission point of the run.

    The aggregated PolarPoint carries ``cl = Σ w_i · CL_i``, so the
    CL floor is rescaled accordingly:

    * mono-point: ``cl_threshold = limits.cl_min * w_primary``
    * multi-point: ``cl_threshold = 0.85 · Σ_i w_i · limits_i.cl_min``
      (slightly under the all-mission sum so the front is feasible
      while still meaningfully demanding).
    """
    primary = next(mp for mp in mission if mp.name == primary_name)
    primary_lim = ENGINEERING_LIMITS.get(primary.name, _DEFAULT_LIMITS)
    if len(mission) == 1:
        cl_threshold = primary_lim["cl_min"] * primary.lift_weight
    else:
        cl_threshold = 0.85 * sum(
            ENGINEERING_LIMITS.get(mp.name, _DEFAULT_LIMITS)["cl_min"] * mp.lift_weight
            for mp in mission
        )
    return [
        MinLiftConstraint(cl_min=cl_threshold),
        MaxDragConstraint(cd_max=primary_lim["cd_max"]),
        MaxAbsPitchingMomentConstraint(cm_abs_max=primary_lim["cm_abs_max"]),
    ]


# ========================================================================== #
# 5. DESIGN-SPACE DECODER
# ========================================================================== #
def build_naca4(params: dict[str, float]) -> Airfoil:
    """Decode three continuous variables into a NACA 4-digit airfoil."""
    m_digit = int(round(float(np.clip(params["m"] * 10.0, 0.0, 9.0))))
    p_digit = int(round(float(np.clip(params["p"] * 10.0, 1.0, 9.0))))
    if m_digit == 0:
        p_digit = 0
    t_pct = int(round(float(np.clip(params["t"] * 30.0 + 6.0, 6.0, 36.0))))
    return NACA4Generator(f"{m_digit}{p_digit}{t_pct:02d}", n_points=120).generate()


DESIGN_SPACE = DesignSpace(
    [
        DesignVariable("m", 0.00, 0.95, label=VAR["m"][1]),
        DesignVariable("p", 0.20, 0.80, label=VAR["p"][1]),
        DesignVariable("t", 0.01, 0.40, label=VAR["t"][1]),
    ]
)


# ========================================================================== #
# 6. HELPERS — classification, bounds (sentinel-aware), hypervolume, knee
# ========================================================================== #
_FAIL_THRESHOLD = 1.0e4  # F values above this are AirfoilEvaluator sentinels


def classify(snap: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (feasible, pareto, infeasible) boolean masks."""
    f = np.asarray(snap.f, dtype=float)
    g = np.asarray(snap.g, dtype=float) if snap.g is not None else np.zeros((f.shape[0], 1))
    feasible = np.all(g <= 1e-9, axis=1)
    # Also flag candidates whose objectives are sentinel-large as infeasible.
    finite_f = np.all(np.abs(f) < _FAIL_THRESHOLD, axis=1)
    feasible = feasible & finite_f
    pareto = np.zeros(f.shape[0], dtype=bool)
    if np.any(feasible):
        mask_local = non_dominated_mask(f[feasible])
        pareto[np.where(feasible)[0][mask_local]] = True
    return feasible, pareto, ~feasible


def axis_bounds_objective(history: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute objective-space axis bounds excluding sentinel-failure values."""
    all_f = np.vstack([np.asarray(s.f, dtype=float) for s in history.snapshots])
    finite = np.all(np.abs(all_f) < _FAIL_THRESHOLD, axis=1)
    if not np.any(finite):
        return (0.0, 1.0), (0.0, 1.0)
    f = all_f[finite]
    cl = -f[:, 0]
    cd = f[:, 1]

    def pad(lo: float, hi: float) -> tuple[float, float]:
        m = 0.06 * (hi - lo + 1e-12)
        return lo - m, hi + m

    return pad(float(cd.min()), float(cd.max())), pad(float(cl.min()), float(cl.max()))


def axis_bounds_design(history: Any) -> dict[str, tuple[float, float]]:
    """Design-space bounds from the design variables themselves (always valid)."""
    return {
        "m": (DESIGN_SPACE.variables[0].lower, DESIGN_SPACE.variables[0].upper),
        "p": (DESIGN_SPACE.variables[1].lower, DESIGN_SPACE.variables[1].upper),
        "t": (DESIGN_SPACE.variables[2].lower, DESIGN_SPACE.variables[2].upper),
    }


def hypervolume_history(history: Any) -> np.ndarray:
    """Compute hypervolume per generation against a fixed worst-case ref point."""
    # Reference point = worst feasible (CD, -CL) ever seen + margin.
    all_finite_f = []
    for s in history.snapshots:
        f = np.asarray(s.f, dtype=float)
        feas, _, _ = classify(s)
        if np.any(feas):
            all_finite_f.append(f[feas])
    if not all_finite_f:
        return np.zeros(len(history.snapshots))
    stack = np.vstack(all_finite_f)
    ref = np.array([stack[:, 0].max(), stack[:, 1].max()]) + 0.1 * np.abs(
        np.array([stack[:, 0].max(), stack[:, 1].max()])
    )
    # Compute HV per generation against this ref point.
    try:
        from pymoo.indicators.hv import HV  # type: ignore[import-not-found]
    except ImportError:
        return np.zeros(len(history.snapshots))
    hv_indicator = HV(ref_point=ref)
    hv = []
    for s in history.snapshots:
        feas, par, _ = classify(s)
        f = np.asarray(s.f, dtype=float)
        if np.any(par):
            hv.append(float(hv_indicator(f[par])))
        else:
            hv.append(0.0)
    return np.asarray(hv, dtype=float)


def knee_point_index(f_pareto: np.ndarray) -> int:
    """Knee-point (closest-to-ideal) index in a normalised 2-objective Pareto.

    The recommended design is the Pareto-optimal solution that is closest
    (in normalised L2 distance) to the utopia point (0, 0) of the
    [F1, F2] Pareto-front bounding box. This is the classic compromise
    solution: it balances both objectives without committing to either
    extreme.
    """
    if f_pareto.shape[0] == 1:
        return 0
    f1 = f_pareto[:, 0]
    f2 = f_pareto[:, 1]
    span1 = max(f1.max() - f1.min(), 1.0e-12)
    span2 = max(f2.max() - f2.min(), 1.0e-12)
    n = np.stack([(f1 - f1.min()) / span1, (f2 - f2.min()) / span2], axis=1)
    dist = np.linalg.norm(n, axis=1)
    return int(np.argmin(dist))


def recommended_index_in_population(snap: Any) -> int | None:
    """Return the index (in the snapshot population) of the recommended design."""
    feas, par, _ = classify(snap)
    if not np.any(par):
        return None
    f = np.asarray(snap.f, dtype=float)
    par_idx = np.where(par)[0]
    knee = knee_point_index(f[par_idx])
    return int(par_idx[knee])


# ========================================================================== #
# 7. FRAME RASTERISATION + GIF WRITER (identical frame sizes guaranteed)
# ========================================================================== #
def _fig_to_rgb(fig: Any) -> np.ndarray:
    """Rasterise without bbox_inches="tight" — exact figsize × dpi pixels."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return np.asarray(imageio.imread(buf))


def _save_gif(frames: list[np.ndarray], output: Path, fps: int) -> Path:
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        h_min = min(s[0] for s in shapes)
        w_min = min(s[1] for s in shapes)
        frames = [f[:h_min, :w_min] for f in frames]
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, duration=1.0 / fps, loop=0)
    return output


# ========================================================================== #
# 8. CUMULATIVE ACCUMULATOR — keeps explored points across generations
# ========================================================================== #
class CumulativeAccumulator:
    """Buckets cumulative (infeasible, dominated-feasible, past-Pareto) points."""

    def __init__(self) -> None:
        self.inf_x: list[np.ndarray] = []
        self.dom_x: list[np.ndarray] = []
        self.par_x: list[np.ndarray] = []
        self.inf_f: list[np.ndarray] = []
        self.dom_f: list[np.ndarray] = []
        self.par_f: list[np.ndarray] = []

    def update(self, snap: Any) -> None:
        f = np.asarray(snap.f, dtype=float)
        x = np.asarray(snap.x, dtype=float)
        feas, par, inf = classify(snap)
        dom = feas & ~par
        self.inf_x.append(x[inf])
        self.dom_x.append(x[dom])
        self.par_x.append(x[par])
        self.inf_f.append(f[inf])
        self.dom_f.append(f[dom])
        self.par_f.append(f[par])

    def _cum(self, lst: list[np.ndarray], upto: int, n_cols: int) -> np.ndarray:
        if not lst[:upto]:
            return np.empty((0, n_cols))
        return np.concatenate(lst[:upto], axis=0)

    def cum_inf_x(self, upto: int) -> np.ndarray:
        return self._cum(self.inf_x, upto, 3)

    def cum_dom_x(self, upto: int) -> np.ndarray:
        return self._cum(self.dom_x, upto, 3)

    def cum_par_x(self, upto: int) -> np.ndarray:
        return self._cum(self.par_x, upto, 3)

    def cum_inf_f(self, upto: int) -> np.ndarray:
        return self._cum(self.inf_f, upto, 2)

    def cum_dom_f(self, upto: int) -> np.ndarray:
        return self._cum(self.dom_f, upto, 2)

    def cum_par_f(self, upto: int) -> np.ndarray:
        return self._cum(self.par_f, upto, 2)


# ========================================================================== #
# 9. STANDARD LEGEND (always exactly 4-5 entries — never one per profile)
# ========================================================================== #
def standard_population_legend(ax: Any, with_recommended: bool = True) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="x",
            color=PALETTE["infeasible"],
            lw=0,
            markersize=7,
            label="Infeasible (current generation)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=PALETTE["feasible"],
            lw=0,
            markersize=6,
            label="Feasible — dominated (current)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=PALETTE["pareto"],
            lw=0,
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label="Pareto front (current)",
        ),
        Patch(
            facecolor=PALETTE["ghost_feas"], edgecolor="none", label="Explored — past generations"
        ),
    ]
    if with_recommended:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color=PALETTE["recommended"],
                lw=0,
                markersize=13,
                markeredgecolor="white",
                markeredgewidth=1.0,
                label="Recommended design",
            )
        )
    ax.legend(handles=handles, loc="best", fontsize=8.6)


# ========================================================================== #
# 10. SCATTER PRIMITIVES (objective and design space)
# ========================================================================== #
def draw_objective(
    ax: Any,
    acc: CumulativeAccumulator,
    snap: Any,
    upto: int,
    obj_x: tuple,
    obj_y: tuple,
    f1_label: str,
    f2_label: str,
    show_legend: bool = True,
) -> None:
    past_inf, past_dom, past_par = acc.cum_inf_f(upto), acc.cum_dom_f(upto), acc.cum_par_f(upto)
    if past_inf.size:
        ax.scatter(
            past_inf[:, 1], -past_inf[:, 0], s=11, c=PALETTE["ghost_inf"], alpha=0.55, marker="x"
        )
    if past_dom.size:
        ax.scatter(past_dom[:, 1], -past_dom[:, 0], s=11, c=PALETTE["ghost_feas"], alpha=0.55)
    if past_par.size:
        ax.scatter(past_par[:, 1], -past_par[:, 0], s=14, c=PALETTE["ghost_par"], alpha=0.6)
    f = np.asarray(snap.f, dtype=float)
    feas, par, inf = classify(snap)
    dom = feas & ~par
    if np.any(inf):
        # Don't draw sentinel-failed points outside the axis range
        finite = np.abs(f).max(axis=1) < _FAIL_THRESHOLD
        plot_inf = inf & finite
        if np.any(plot_inf):
            ax.scatter(
                f[plot_inf, 1],
                -f[plot_inf, 0],
                s=34,
                c=PALETTE["infeasible"],
                alpha=0.85,
                marker="x",
                lw=1.4,
            )
    if np.any(dom):
        ax.scatter(f[dom, 1], -f[dom, 0], s=30, c=PALETTE["feasible"], alpha=0.85)
    if np.any(par):
        ax.scatter(
            f[par, 1], -f[par, 0], s=80, c=PALETTE["pareto"], edgecolors="white", lw=1.1, zorder=3
        )
    # Highlight the recommended design
    rec_idx = recommended_index_in_population(snap)
    if rec_idx is not None:
        ax.scatter(
            [f[rec_idx, 1]],
            [-f[rec_idx, 0]],
            s=240,
            marker="*",
            c=PALETTE["recommended"],
            edgecolors="white",
            lw=1.2,
            zorder=4,
        )
    ax.set_xlim(*obj_x)
    ax.set_ylim(*obj_y)
    ax.set_xlabel(f1_label)
    ax.set_ylabel(f2_label)
    ax.set_title(f"Objective space — Generation {snap.generation + 1}")
    if show_legend:
        standard_population_legend(ax)


_DESIGN_PROJ = (("m", "t", 0, 2), ("p", "t", 1, 2), ("m", "p", 0, 1))


def draw_design_proj(
    ax: Any,
    acc: CumulativeAccumulator,
    snap: Any,
    upto: int,
    db: dict[str, tuple],
    lx: str,
    ly: str,
    ix: int,
    iy: int,
) -> None:
    past_inf, past_dom, past_par = acc.cum_inf_x(upto), acc.cum_dom_x(upto), acc.cum_par_x(upto)
    if past_inf.size:
        ax.scatter(
            past_inf[:, ix], past_inf[:, iy], s=10, c=PALETTE["ghost_inf"], alpha=0.5, marker="x"
        )
    if past_dom.size:
        ax.scatter(past_dom[:, ix], past_dom[:, iy], s=10, c=PALETTE["ghost_feas"], alpha=0.5)
    if past_par.size:
        ax.scatter(past_par[:, ix], past_par[:, iy], s=12, c=PALETTE["ghost_par"], alpha=0.55)
    x = np.asarray(snap.x, dtype=float)
    feas, par, inf = classify(snap)
    dom = feas & ~par
    if np.any(inf):
        ax.scatter(
            x[inf, ix], x[inf, iy], s=30, c=PALETTE["infeasible"], alpha=0.85, marker="x", lw=1.3
        )
    if np.any(dom):
        ax.scatter(x[dom, ix], x[dom, iy], s=26, c=PALETTE["feasible"], alpha=0.85)
    if np.any(par):
        ax.scatter(
            x[par, ix], x[par, iy], s=70, c=PALETTE["pareto"], edgecolors="white", lw=1.0, zorder=3
        )
    rec_idx = recommended_index_in_population(snap)
    if rec_idx is not None:
        ax.scatter(
            [x[rec_idx, ix]],
            [x[rec_idx, iy]],
            s=240,
            marker="*",
            c=PALETTE["recommended"],
            edgecolors="white",
            lw=1.2,
            zorder=4,
        )
    ax.set_xlim(*db[lx])
    ax.set_ylim(*db[ly])
    ax.set_xlabel(axis_label(lx))
    ax.set_ylabel(axis_label(ly))
    ax.set_title(f"Design space — {VAR[lx][1]} vs {VAR[ly][1]}")


# ========================================================================== #
# 11. PARALLEL COORDINATES — with real min/mid/max tick labels
# ========================================================================== #
def parallel_var_bounds(history: Any) -> list[tuple[float, float, str, str]]:
    """Return list of (lo, hi, axis_label, format) tuples for each parallel axis."""
    all_x = np.vstack([np.asarray(s.x, dtype=float) for s in history.snapshots])
    all_f = np.vstack([np.asarray(s.f, dtype=float) for s in history.snapshots])
    finite_f = np.all(np.abs(all_f) < _FAIL_THRESHOLD, axis=1)
    f_ok = all_f[finite_f] if np.any(finite_f) else all_f
    return [
        (float(all_x[:, 0].min()), float(all_x[:, 0].max()), VAR["m"][1], "{:.2f}"),
        (float(all_x[:, 1].min()), float(all_x[:, 1].max()), VAR["p"][1], "{:.2f}"),
        (float(all_x[:, 2].min()), float(all_x[:, 2].max()), VAR["t"][1], "{:.2f}"),
        (float((-f_ok[:, 0]).min()), float((-f_ok[:, 0]).max()), VAR["cl_tot"][1], "{:.2f}"),
        (float(f_ok[:, 1].min()), float(f_ok[:, 1].max()), VAR["cd_cr"][1], "{:.4f}"),
    ]


def draw_parallel(
    ax: Any,
    acc: CumulativeAccumulator,
    snap: Any,
    upto: int,
    var_bounds: list[tuple[float, float, str, str]],
) -> None:
    n_axes = len(var_bounds)
    xs = np.arange(n_axes)

    def normalise(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
        if hi - lo < 1e-12:
            return np.full_like(values, 0.5, dtype=float)
        return np.clip((values - lo) / (hi - lo), -0.05, 1.05)

    def stack(x: np.ndarray, f: np.ndarray) -> np.ndarray:
        cols = [
            normalise(x[:, 0], var_bounds[0][0], var_bounds[0][1]),
            normalise(x[:, 1], var_bounds[1][0], var_bounds[1][1]),
            normalise(x[:, 2], var_bounds[2][0], var_bounds[2][1]),
            normalise(-f[:, 0], var_bounds[3][0], var_bounds[3][1]),
            normalise(f[:, 1], var_bounds[4][0], var_bounds[4][1]),
        ]
        return np.stack(cols, axis=1)

    past_inf_x, past_inf_f = acc.cum_inf_x(upto), acc.cum_inf_f(upto)
    past_dom_x, past_dom_f = acc.cum_dom_x(upto), acc.cum_dom_f(upto)
    if past_inf_x.size:
        finite = np.all(np.abs(past_inf_f) < _FAIL_THRESHOLD, axis=1)
        for r in stack(past_inf_x[finite], past_inf_f[finite]):
            ax.plot(xs, r, color=PALETTE["ghost_inf"], lw=0.55, alpha=0.45)
    if past_dom_x.size:
        for r in stack(past_dom_x, past_dom_f):
            ax.plot(xs, r, color=PALETTE["ghost_feas"], lw=0.55, alpha=0.4)
    x = np.asarray(snap.x, dtype=float)
    f = np.asarray(snap.f, dtype=float)
    feas, par, inf = classify(snap)
    dom = feas & ~par
    finite_pop = np.all(np.abs(f) < _FAIL_THRESHOLD, axis=1)
    if np.any(inf & finite_pop):
        for r in stack(x[inf & finite_pop], f[inf & finite_pop]):
            ax.plot(xs, r, color=PALETTE["infeasible"], lw=0.85, alpha=0.65)
    if np.any(dom):
        for r in stack(x[dom], f[dom]):
            ax.plot(xs, r, color=PALETTE["feasible"], lw=0.95, alpha=0.7)
    if np.any(par):
        for r in stack(x[par], f[par]):
            ax.plot(xs, r, color=PALETTE["pareto"], lw=1.7, alpha=0.95)
    # Recommended in deep navy on top
    rec_idx = recommended_index_in_population(snap)
    if rec_idx is not None:
        rec_row = stack(x[rec_idx : rec_idx + 1], f[rec_idx : rec_idx + 1])[0]
        ax.plot(xs, rec_row, color=PALETTE["recommended"], lw=2.6, alpha=1.0, zorder=5)

    # X tick labels = "symbol  (description)"
    ax.set_xticks(xs)
    tick_lbls = []
    for i, (_lo, _hi, desc, _fmt) in enumerate(var_bounds):
        sym_keys = ["m", "p", "t", "cl_tot", "cd_cr"]
        sym = VAR[sym_keys[i]][0]
        tick_lbls.append(f"{sym}\n({desc})")
    ax.set_xticklabels(tick_lbls, fontsize=9.0)
    # Show actual numeric bounds vertically next to each axis with min / mid / max
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["", "", ""])
    ax.set_ylim(-0.07, 1.07)
    for i, (lo, hi, _desc, fmt) in enumerate(var_bounds):
        mid = (lo + hi) / 2.0
        ax.text(i, -0.05, fmt.format(lo), ha="center", va="top", fontsize=8.2, color="0.3")
        ax.text(
            i,
            0.5,
            fmt.format(mid),
            ha="center",
            va="center",
            fontsize=8.2,
            color="0.45",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.7},
        )
        ax.text(i, 1.05, fmt.format(hi), ha="center", va="bottom", fontsize=8.2, color="0.3")
    ax.grid(alpha=0.25, ls="--", axis="y")
    ax.set_title(f"Parallel coordinates — Generation {snap.generation + 1}")


# ========================================================================== #
# 12. PER-RUN RENDERERS (asset gallery for ONE optimisation)
# ========================================================================== #
def render_pareto_evolution(
    history: Any,
    out_gif: Path,
    out_png: Path,
    fps: int,
    f1_label: str,
    f2_label: str,
) -> None:
    obj_x, obj_y = axis_bounds_objective(history)
    db = axis_bounds_design(history)
    acc = CumulativeAccumulator()
    figsize = (12.8, 5.0)
    frames: list[np.ndarray] = []
    for k, snap in enumerate(history.snapshots):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.0], wspace=0.32)
        ax_obj, ax_m, ax_p = (fig.add_subplot(gs[0, i]) for i in range(3))
        draw_objective(ax_obj, acc, snap, k, obj_x, obj_y, f1_label, f2_label)
        draw_design_proj(ax_m, acc, snap, k, db, "m", "t", 0, 2)
        draw_design_proj(ax_p, acc, snap, k, db, "p", "t", 1, 2)
        fig.suptitle(
            f"Generation {snap.generation + 1} of {len(history.snapshots)}",
            fontsize=13,
            fontweight="semibold",
            y=0.99,
        )
        fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.13, wspace=0.32)
        frames.append(_fig_to_rgb(fig))
        acc.update(snap)
    _save_gif(frames, out_gif, fps)
    # Final state PNG (rebuild last frame)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.0], wspace=0.32)
    ax_obj, ax_m, ax_p = (fig.add_subplot(gs[0, i]) for i in range(3))
    last = history.snapshots[-1]
    draw_objective(ax_obj, acc, last, len(history.snapshots) - 1, obj_x, obj_y, f1_label, f2_label)
    draw_design_proj(ax_m, acc, last, len(history.snapshots) - 1, db, "m", "t", 0, 2)
    draw_design_proj(ax_p, acc, last, len(history.snapshots) - 1, db, "p", "t", 1, 2)
    fig.suptitle("Final state — Pareto evolution", fontsize=13, fontweight="semibold", y=0.99)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def render_design_evolution(
    history: Any,
    out_gif: Path,
    out_png: Path,
    fps: int,
) -> None:
    db = axis_bounds_design(history)
    acc = CumulativeAccumulator()
    figsize = (13.0, 4.4)
    frames: list[np.ndarray] = []
    for k, snap in enumerate(history.snapshots):
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, (lx, ly, ix, iy) in zip(axes, _DESIGN_PROJ, strict=True):
            draw_design_proj(ax, acc, snap, k, db, lx, ly, ix, iy)
        standard_population_legend(axes[0])
        fig.suptitle(
            f"Design space — Generation {snap.generation + 1}",
            fontsize=13,
            fontweight="semibold",
            y=0.99,
        )
        fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.14, wspace=0.3)
        frames.append(_fig_to_rgb(fig))
        acc.update(snap)
    _save_gif(frames, out_gif, fps)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    last = history.snapshots[-1]
    for ax, (lx, ly, ix, iy) in zip(axes, _DESIGN_PROJ, strict=True):
        draw_design_proj(ax, acc, last, len(history.snapshots) - 1, db, lx, ly, ix, iy)
    standard_population_legend(axes[0])
    fig.suptitle("Final state — design space", fontsize=13, fontweight="semibold", y=0.99)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.14, wspace=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def render_parallel_evolution(
    history: Any,
    out_gif: Path,
    out_png: Path,
    fps: int,
) -> None:
    vb = parallel_var_bounds(history)
    acc = CumulativeAccumulator()
    figsize = (11.5, 4.8)
    frames: list[np.ndarray] = []
    for k, snap in enumerate(history.snapshots):
        fig, ax = plt.subplots(figsize=figsize)
        draw_parallel(ax, acc, snap, k, vb)
        standard_population_legend(ax)
        fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=0.16)
        frames.append(_fig_to_rgb(fig))
        acc.update(snap)
    _save_gif(frames, out_gif, fps)
    fig, ax = plt.subplots(figsize=figsize)
    draw_parallel(ax, acc, history.snapshots[-1], len(history.snapshots) - 1, vb)
    standard_population_legend(ax)
    ax.set_title(
        f"Final state — parallel coordinates (Generation {history.snapshots[-1].generation + 1})"
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=0.16)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def render_geometry_evolution(
    history: Any,
    evaluator: AirfoilEvaluator,
    out_gif: Path,
    out_png: Path,
    fps: int,
    fade_window: int,
) -> None:
    # Pre-decode Pareto airfoils per gen + recommended airfoil per gen.
    per_gen_pareto: list[list[Airfoil]] = []
    per_gen_recommended: list[Airfoil | None] = []
    for snap in history.snapshots:
        _, par, _ = classify(snap)
        airfoils: list[Airfoil] = []
        if np.any(par):
            x = np.asarray(snap.x, dtype=float)
            for idx in np.where(par)[0]:
                try:
                    airfoils.append(evaluator.genome_to_airfoil(x[idx]))
                except Exception:  # noqa: BLE001
                    continue
        per_gen_pareto.append(airfoils)
        rec_idx = recommended_index_in_population(snap)
        if rec_idx is not None:
            try:
                per_gen_recommended.append(
                    evaluator.genome_to_airfoil(np.asarray(snap.x, dtype=float)[rec_idx])
                )
            except Exception:  # noqa: BLE001
                per_gen_recommended.append(None)
        else:
            per_gen_recommended.append(None)
    # Y-axis bounds
    all_y: list[float] = []
    for ag in per_gen_pareto:
        for af in ag:
            all_y.extend([float(af.y.min()), float(af.y.max())])
    y_lo, y_hi = (min(all_y) - 0.02, max(all_y) + 0.02) if all_y else (-0.1, 0.1)

    figsize = (8.6, 4.8)
    frames: list[np.ndarray] = []

    def render_frame(i: int) -> None:
        # Faded past geometries (only the last `fade_window` generations).
        start = max(0, i - fade_window)
        # Past Pareto geometries
        for j in range(start, i):
            for af in per_gen_pareto[j]:
                ax.plot(af.x, af.y, color=PALETTE["geom_history"], lw=0.6, alpha=0.4, zorder=1)
        # Current generation Pareto in amber
        for af in per_gen_pareto[i]:
            ax.plot(af.x, af.y, color=PALETTE["pareto"], lw=1.7, alpha=0.4, zorder=2)
        # Recommended geometry overlaid in deep navy
        if per_gen_recommended[i] is not None:
            af = per_gen_recommended[i]
            ax.plot(af.x, af.y, color=PALETTE["recommended"], lw=2.4, alpha=1.0, zorder=3)
            ax.text(
                0.99,
                0.97,
                f"Recommended: {af.name}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10.0,
                color=PALETTE["recommended"],
                bbox={"facecolor": "white", "edgecolor": "0.7", "pad": 2.5, "alpha": 0.92},
            )

    for i, _ in enumerate(per_gen_pareto):
        fig, ax = plt.subplots(figsize=figsize)
        render_frame(i)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(r"$x/c$  (Chordwise position)")
        ax.set_ylabel(r"$y/c$  (Vertical coordinate)")
        ax.set_title(f"Pareto geometries — Generation {history.snapshots[i].generation + 1}")
        # 2-entry legend ONLY (no per-airfoil names)
        legend_handles = [
            Line2D([0], [0], color=PALETTE["pareto"], lw=1.7, label="Current Pareto geometries"),
            Line2D(
                [0],
                [0],
                color=PALETTE["geom_history"],
                lw=0.8,
                label=f"Past {fade_window} generations",
            ),
        ]
        if per_gen_recommended[i] is not None:
            legend_handles.append(
                Line2D([0], [0], color=PALETTE["recommended"], lw=2.4, label="Recommended design")
            )
        ax.legend(handles=legend_handles, loc="lower right", fontsize=9.0)
        fig.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=0.13)
        frames.append(_fig_to_rgb(fig))
    _save_gif(frames, out_gif, fps)
    # Final PNG
    fig, ax = plt.subplots(figsize=figsize)
    i = len(per_gen_pareto) - 1
    render_frame(i)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r"$x/c$  (Chordwise position)")
    ax.set_ylabel(r"$y/c$  (Vertical coordinate)")
    ax.set_title("Final state — Pareto-optimal geometries")
    legend_handles = [
        Line2D([0], [0], color=PALETTE["pareto"], lw=1.7, label="Final Pareto geometries"),
        Line2D(
            [0],
            [0],
            color=PALETTE["geom_history"],
            lw=0.8,
            label=f"Past {fade_window} generations",
        ),
    ]
    if per_gen_recommended[i] is not None:
        legend_handles.append(
            Line2D([0], [0], color=PALETTE["recommended"], lw=2.4, label="Recommended design")
        )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9.0)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=0.13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def render_convergence(history: Any, out_png: Path) -> None:
    n_gen = len(history.snapshots)
    best_f1, best_f2, par_size = [], [], []
    for snap in history.snapshots:
        feas, par, _ = classify(snap)
        f = np.asarray(snap.f, dtype=float)
        if np.any(feas):
            best_f1.append(float((-f[feas, 0]).max()))
            best_f2.append(float(f[feas, 1].min()))
        else:
            best_f1.append(np.nan)
            best_f2.append(np.nan)
        par_size.append(int(par.sum()))
    hv = hypervolume_history(history)
    gens = np.arange(1, n_gen + 1)
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.9))
    axes[0].plot(gens, best_f1, "-o", color=PALETTE["recommended"], ms=4)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best feasible " + VAR["cl_tot"][1])
    axes[0].set_title("Lift convergence")
    axes[1].plot(gens, best_f2, "-o", color=PALETTE["infeasible"], ms=4)
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Best feasible " + VAR["cd_cr"][1])
    axes[1].set_title("Drag convergence")
    axes[2].plot(gens, par_size, "-o", color=PALETTE["pareto"], ms=4)
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("|Pareto set|")
    axes[2].set_title("Pareto-set size")
    axes[3].plot(gens, hv, "-o", color="#196b3a", ms=4)
    axes[3].set_xlabel("Generation")
    axes[3].set_ylabel("Hypervolume (HV)")
    axes[3].set_title("Hypervolume convergence")
    for ax in axes:
        ax.grid(alpha=0.3, ls="--")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.16, wspace=0.32)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def render_mission_breakdown(
    history: Any,
    evaluator: AirfoilEvaluator,
    solver: AbstractSolver,
    out_png: Path,
    mission: tuple[MissionPoint, ...],
) -> None:
    snap = history.snapshots[-1]
    _, par, _ = classify(snap)
    if not np.any(par):
        return
    x = np.asarray(snap.x, dtype=float)
    par_idx = np.where(par)[0]
    rec_idx = recommended_index_in_population(snap)
    airfoils: list[tuple[Airfoil, bool]] = []  # (airfoil, is_recommended)
    for idx in par_idx:
        try:
            af = evaluator.genome_to_airfoil(x[idx])
            airfoils.append((af, idx == rec_idx))
        except Exception:  # noqa: BLE001
            continue
    if not airfoils:
        return
    cl_table = np.zeros((len(airfoils), len(mission)))
    cd_table = np.zeros((len(airfoils), len(mission)))
    for i, (af, _) in enumerate(airfoils):
        results = per_point_analysis(solver, af, mission)
        for j, mp in enumerate(mission):
            cl_table[i, j], cd_table[i, j], _ = results[mp.name]
    ld_table = cl_table / np.maximum(cd_table, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    x_axis = np.arange(len(mission))
    width = 0.78 / max(len(airfoils), 1)
    for i, (_af, is_rec) in enumerate(airfoils):
        xs = x_axis + (i - len(airfoils) / 2.0) * width + width / 2.0
        c = PALETTE["recommended"] if is_rec else PALETTE["pareto"]
        edge = "black" if is_rec else "white"
        lw = 1.2 if is_rec else 0.55
        alpha = 1.0 if is_rec else 0.55
        for ax, data in zip(axes, (cl_table[i], cd_table[i], ld_table[i]), strict=True):
            ax.bar(xs, data, width=width, color=c, edgecolor=edge, lw=lw, alpha=alpha)
    titles_units = [
        (axis_label("cl"), r"$C_l$ across mission points"),
        (axis_label("cd"), r"$C_d$ across mission points"),
        (axis_label("ld"), r"$L/D$ across mission points"),
    ]
    for ax, (ylab, ttl) in zip(axes, titles_units, strict=True):
        ax.set_xticks(x_axis)
        ax.set_xticklabels([mp.pretty for mp in mission])
        ax.set_xlabel("Mission point")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(alpha=0.25, ls="--", axis="y")
    legend_handles = [
        Patch(
            facecolor=PALETTE["pareto"],
            edgecolor="white",
            alpha=0.55,
            label="Pareto-optimal candidates",
        ),
        Patch(
            facecolor=PALETTE["recommended"], edgecolor="black", lw=1.2, label="Recommended design"
        ),
    ]
    axes[0].legend(handles=legend_handles, loc="best", fontsize=9.0)
    fig.suptitle(
        "Mission breakdown — performance of every Pareto airfoil at each leg",
        fontsize=12.5,
        fontweight="semibold",
        y=0.99,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.14, wspace=0.32)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def render_multipoint_polars(
    history: Any,
    evaluator: AirfoilEvaluator,
    solver: AbstractSolver,
    out_png: Path,
    mission: tuple[MissionPoint, ...],
) -> None:
    snap = history.snapshots[-1]
    feas, par, _ = classify(snap)
    if not np.any(par):
        return
    f = np.asarray(snap.f, dtype=float)
    x = np.asarray(snap.x, dtype=float)
    par_idx = np.where(par)[0]
    par_f = f[par_idx]
    cl = -par_f[:, 0]
    cd = par_f[:, 1]
    idx_maxcl = par_idx[int(np.argmax(cl))]
    idx_mincd = par_idx[int(np.argmin(cd))]
    rec_in_pop = recommended_index_in_population(snap)
    rec_idx = rec_in_pop if rec_in_pop is not None else int(par_idx[knee_point_index(par_f)])
    # Labels: Max Lift, Min Drag, Recommended (no "knee")
    extrema = {
        "Max-Lift Pareto": (idx_maxcl, PALETTE["mission"][1], "-", "o"),
        "Min-Drag Pareto": (idx_mincd, PALETTE["mission"][2], "-", "s"),
        "Recommended": (rec_idx, PALETTE["recommended"], "-", "*"),
    }
    n_cols = len(mission)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.2 * n_cols + 0.4, 7.6), squeeze=False)
    for col, mp in enumerate(mission):
        alphas = np.linspace(mp.alpha - 4.0, mp.alpha + 6.0, 24)
        for label, (idx, color, ls, marker) in extrema.items():
            try:
                af = evaluator.genome_to_airfoil(x[idx])
            except Exception:  # noqa: BLE001
                continue
            cls, cds = [], []
            for a in alphas:
                try:
                    c_l, c_d, _ = solver.analyze_single(  # type: ignore[attr-defined]
                        af, float(a), mp.reynolds, mp.mach
                    )
                    cls.append(c_l)
                    cds.append(c_d)
                except Exception:  # noqa: BLE001
                    cls.append(np.nan)
                    cds.append(np.nan)
            ms = 5.0 if label == "Recommended" else 3.2
            lw = 2.0 if label == "Recommended" else 1.5
            axes[0, col].plot(alphas, cls, ls + marker, color=color, ms=ms, lw=lw, label=label)
            axes[1, col].plot(cds, cls, ls + marker, color=color, ms=ms, lw=lw, label=label)
        axes[0, col].axvline(mp.alpha, color="0.55", ls=":", lw=0.85)
        axes[0, col].set_title(
            f"{mp.pretty}: "
            r"$Re$"
            f" = {mp.reynolds:.0e}, "
            r"$M$"
            f" = {mp.mach:.2f}, target "
            r"$\alpha$"
            f" = {mp.alpha:.0f}"
            r"$^\circ$"
        )
        axes[0, col].set_xlabel(axis_label("alpha", "deg"))
        axes[0, col].set_ylabel(axis_label("cl"))
        axes[1, col].set_xlabel(axis_label("cd"))
        axes[1, col].set_ylabel(axis_label("cl"))
        axes[0, col].legend(loc="best", fontsize=8.5)
    fig.suptitle(
        "Multi-point polars — Max-Lift, Min-Drag, and Recommended Pareto designs",
        fontsize=12.5,
        fontweight="semibold",
        y=0.99,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.08, wspace=0.32, hspace=0.32)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def render_pareto_geometries(
    history: Any,
    evaluator: AirfoilEvaluator,
    out_png: Path,
) -> None:
    snap = history.snapshots[-1]
    _, par, _ = classify(snap)
    rec_idx = recommended_index_in_population(snap)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.asarray(snap.x, dtype=float)
    n_pareto = 0
    rec_name: str | None = None
    for idx in np.where(par)[0]:
        try:
            af = evaluator.genome_to_airfoil(x[idx])
        except Exception:  # noqa: BLE001
            continue
        if idx == rec_idx:
            ax.plot(af.x, af.y, color=PALETTE["recommended"], lw=2.4, alpha=1.0, zorder=4)
            rec_name = af.name
        else:
            ax.plot(af.x, af.y, color=PALETTE["pareto"], lw=1.2, alpha=0.6, zorder=2)
        n_pareto += 1
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x/c$  (Chordwise position)")
    ax.set_ylabel(r"$y/c$  (Vertical coordinate)")
    ax.set_title(f"Final Pareto geometries ({n_pareto} candidates)")
    legend_handles = [
        Line2D(
            [0], [0], color=PALETTE["pareto"], lw=1.2, alpha=0.7, label="Pareto-optimal geometry"
        ),
    ]
    if rec_name is not None:
        legend_handles.append(
            Line2D(
                [0], [0], color=PALETTE["recommended"], lw=2.4, label=f"Recommended ({rec_name})"
            )
        )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9.0)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=0.13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# ========================================================================== #
# 13. ONE FULL PER-RUN ASSET GENERATION
# ========================================================================== #
@dataclass
class RunResult:
    """Bundle of one optimisation's outputs for downstream comparison."""

    name: str  # short ID: "stage1_takeoff", "stage2_multipoint"
    pretty: str  # display: "Stage 1 — Take-off only"
    history: Any
    evaluator: AirfoilEvaluator
    solver: AbstractSolver
    mission: tuple[MissionPoint, ...]  # the missions this run aggregated over
    primary_name: str
    recommended_airfoil: Airfoil | None = None


def render_full_asset_set(
    run: RunResult,
    out_dir: Path,
    fps: int,
    fade_window: int,
    f1_label: str,
    f2_label: str,
) -> None:
    """Generate every PNG + GIF + final-state PNG for one optimisation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  rendering {run.pretty} → {out_dir.resolve()}")
    render_pareto_evolution(
        run.history,
        out_dir / "pareto_evolution.gif",
        out_dir / "pareto_evolution_final.png",
        fps,
        f1_label,
        f2_label,
    )
    render_design_evolution(
        run.history,
        out_dir / "design_evolution.gif",
        out_dir / "design_evolution_final.png",
        fps,
    )
    render_parallel_evolution(
        run.history,
        out_dir / "parallel_evolution.gif",
        out_dir / "parallel_evolution_final.png",
        fps,
    )
    render_geometry_evolution(
        run.history,
        run.evaluator,
        out_dir / "geometry_evolution.gif",
        out_dir / "geometry_evolution_final.png",
        fps,
        fade_window,
    )
    render_convergence(run.history, out_dir / "convergence.png")
    render_pareto_geometries(run.history, run.evaluator, out_dir / "pareto_geometries.png")
    render_mission_breakdown(
        run.history,
        run.evaluator,
        run.solver,
        out_dir / "mission_breakdown.png",
        MISSION,  # always against the full mission profile, even for mono-point
    )
    render_multipoint_polars(
        run.history,
        run.evaluator,
        run.solver,
        out_dir / "multipoint_polars.png",
        MISSION,
    )


# ========================================================================== #
# 14. CROSS-COMPARISON FIGURES (overlay all run champions)
# ========================================================================== #
def render_comparison_champion_geometries(runs: list[RunResult], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    colors = PALETTE["mission"]
    for i, run in enumerate(runs):
        af = run.recommended_airfoil
        if af is None:
            continue
        ax.plot(
            af.x, af.y, color=colors[i % len(colors)], lw=2.0, label=f"{run.pretty}  ({af.name})"
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x/c$  (Chordwise position)")
    ax.set_ylabel(r"$y/c$  (Vertical coordinate)")
    ax.set_title("Champion geometries — every optimisation strategy")
    ax.legend(loc="lower right", fontsize=9.0)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def render_comparison_mission_performance(runs: list[RunResult], out_png: Path) -> None:
    """Compare each champion's CL / CD / L/D at every mission leg."""
    cl_tab = np.zeros((len(runs), len(MISSION)))
    cd_tab = np.zeros((len(runs), len(MISSION)))
    for i, run in enumerate(runs):
        if run.recommended_airfoil is None:
            continue
        results = per_point_analysis(run.solver, run.recommended_airfoil, MISSION)
        for j, mp in enumerate(MISSION):
            cl_tab[i, j], cd_tab[i, j], _ = results[mp.name]
    ld_tab = cl_tab / np.maximum(cd_tab, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    x_axis = np.arange(len(MISSION))
    width = 0.8 / max(len(runs), 1)
    colors = PALETTE["mission"]
    for i, run in enumerate(runs):
        xs = x_axis + (i - len(runs) / 2.0) * width + width / 2.0
        c = colors[i % len(colors)]
        for ax, data in zip(axes, (cl_tab[i], cd_tab[i], ld_tab[i]), strict=True):
            ax.bar(
                xs,
                data,
                width=width,
                color=c,
                edgecolor="white",
                lw=0.7,
                label=run.pretty if ax is axes[0] else None,
            )
    for ax, (ttl, ylab) in zip(
        axes,
        [
            (r"$C_l$ across mission points", axis_label("cl")),
            (r"$C_d$ across mission points", axis_label("cd")),
            (r"$L/D$ across mission points", axis_label("ld")),
        ],
        strict=True,
    ):
        ax.set_xticks(x_axis)
        ax.set_xticklabels([mp.pretty for mp in MISSION])
        ax.set_xlabel("Mission point")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(alpha=0.25, ls="--", axis="y")
    axes[0].legend(loc="best", fontsize=8.8)
    fig.suptitle(
        "Champion comparison — performance of each optimisation's recommended design",
        fontsize=12.5,
        fontweight="semibold",
        y=0.99,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.14, wspace=0.32)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def render_comparison_design_parameters(runs: list[RunResult], out_png: Path) -> None:
    """Compare each champion's (m, p, t) parameters as grouped bars."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    keys = ("m", "p", "t")
    colors = PALETTE["mission"]
    for j, key in enumerate(keys):
        idx = j
        ax = axes[j]
        for i, run in enumerate(runs):
            if run.recommended_airfoil is None:
                continue
            # Extract parameters by inverting the build_naca4 mapping.
            t_val = float(run.recommended_airfoil.max_thickness)
            c_val = float(run.recommended_airfoil.max_camber)
            p_val = float(run.recommended_airfoil.max_camber_location)
            vals = (c_val, p_val, t_val)
            ax.bar(
                [i],
                [vals[idx]],
                color=colors[i % len(colors)],
                edgecolor="white",
                lw=0.7,
                label=run.pretty if j == 0 else None,
            )
        ax.set_xticks(np.arange(len(runs)))
        ax.set_xticklabels(
            [r.name.replace("stage1_", "").replace("stage2_", "") for r in runs],
            rotation=12,
            fontsize=9.0,
        )
        ax.set_title(f"Champion {VAR[key][1]}")
        ax.set_ylabel(VAR[key][0])
        ax.grid(alpha=0.25, ls="--", axis="y")
    axes[0].legend(loc="best", fontsize=8.6)
    fig.suptitle("Champion design parameters", fontsize=12.5, fontweight="semibold", y=0.99)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.18, wspace=0.32)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def render_comparison_pareto_overlay(runs: list[RunResult], out_png: Path) -> None:
    """Project every run's final Pareto front into (CL_total, CD_primary) space."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = PALETTE["mission"]
    for i, run in enumerate(runs):
        snap = run.history.snapshots[-1]
        _, par, _ = classify(snap)
        f = np.asarray(snap.f, dtype=float)
        if np.any(par):
            cl = -f[par, 0]
            cd = f[par, 1]
            order = np.argsort(cd)
            ax.plot(
                cd[order],
                cl[order],
                "-o",
                color=colors[i % len(colors)],
                ms=4.2,
                lw=1.5,
                label=run.pretty,
            )
            if run.recommended_airfoil is not None:
                rec_idx = recommended_index_in_population(snap)
                if rec_idx is not None:
                    ax.scatter(
                        [f[rec_idx, 1]],
                        [-f[rec_idx, 0]],
                        s=200,
                        marker="*",
                        color=colors[i % len(colors)],
                        edgecolors="black",
                        lw=1.2,
                        zorder=5,
                    )
    ax.set_xlabel("Primary " + axis_label("cd") + "  (aggregated over each run's missions)")
    ax.set_ylabel("Weighted " + axis_label("cl_tot"))
    ax.set_title("Pareto-front overlay — all optimisations (★ = recommended)")
    ax.legend(loc="best", fontsize=9.0)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# ========================================================================== #
# 15. DRIVERS — build solver / evaluator / study, run, capture champion
# ========================================================================== #
def make_solver(
    use_xfoil: bool,
    args: argparse.Namespace,
    mission: tuple[MissionPoint, ...],
    primary_name: str,
) -> AbstractSolver:
    """Build either the real-XFOIL or the closed-form solver.

    XFOIL is the default; ``--synthetic`` (which flips ``use_xfoil`` to
    False) opts into the closed-form physical model. When XFOIL is
    requested but the binary is missing on ``PATH``, the script aborts
    with an explicit message rather than silently downgrading.
    """
    if use_xfoil:
        if shutil.which(args.xfoil_binary) is None:
            raise SystemExit(
                f"XFOIL binary '{args.xfoil_binary}' is not on PATH. "
                "Install it (https://web.mit.edu/drela/Public/web/xfoil/) and "
                "re-run, or pass --synthetic to use the closed-form fallback."
            )
        return AggregatedXfoilSolver(
            mission=mission,
            primary_name=primary_name,
            binary=args.xfoil_binary,
            max_iter=args.max_iter,
            n_crit=args.n_crit,
            repanel=not args.no_repanel,
            timeout_s=args.timeout_s,
            x_trip_upper=args.x_trip_upper,
            x_trip_lower=args.x_trip_lower,
        )
    return AggregatedSyntheticSolver(mission=mission, primary_name=primary_name)


def run_one_optimisation(
    name: str,
    pretty: str,
    solver: AbstractSolver,
    mission: tuple[MissionPoint, ...],
    primary_name: str,
    pop: int,
    n_gen: int,
    seed: int,
) -> RunResult:
    """Construct the evaluator + study, run it, capture the recommended airfoil."""
    evaluator = AirfoilEvaluator(
        design_space=DESIGN_SPACE,
        airfoil_factory=build_naca4,
        solver=solver,
        operating_point=OperatingPoint(alpha=2.0, reynolds=2.0e6, mach=0.6),
        objectives=[MaximizeLift(), MinimizeDrag()],
        geometric_constraints=standard_geometric_constraints(),
        physical_constraints=engineering_physical_constraints(mission, primary_name),
    )
    study = OptimizationStudy(
        evaluator=evaluator,
        algorithm=nsga2(pop_size=pop),
        n_gen=n_gen,
        seed=seed,
    )
    print(f"  Optimising {pretty} (pop={pop}, n_gen={n_gen}, seed={seed}) ...")
    study.run()
    # Pick the recommended airfoil from the final generation.
    snap = study.history.snapshots[-1]
    rec_idx = recommended_index_in_population(snap)
    rec_af: Airfoil | None = None
    if rec_idx is not None:
        try:
            rec_af = evaluator.genome_to_airfoil(np.asarray(snap.x, dtype=float)[rec_idx])
        except Exception:  # noqa: BLE001
            rec_af = None
    return RunResult(
        name=name,
        pretty=pretty,
        history=study.history,
        evaluator=evaluator,
        solver=solver,
        mission=mission,
        primary_name=primary_name,
        recommended_airfoil=rec_af,
    )


# ========================================================================== #
# 16. MAIN — Stage 1 (mono per mission) + Stage 2 (multi) + Stage 3 (comparison)
# ========================================================================== #
def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/assets"))
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--n-gen", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument(
        "--fade-window",
        type=int,
        default=10,
        help="Number of past generations shown faded in the geometry GIF.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force the closed-form aerodynamic model instead of the real XFOIL binary.",
    )
    parser.add_argument("--xfoil-binary", default="xfoil")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--n-crit", type=float, default=9.0)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--no-repanel", action="store_true")
    parser.add_argument("--x-trip-upper", type=float, default=None)
    parser.add_argument("--x-trip-lower", type=float, default=None)
    parser.add_argument(
        "--skip-stage1", action="store_true", help="Skip the mono-point stage (faster)."
    )
    parser.add_argument("--skip-stage2", action="store_true", help="Skip the multi-point stage.")
    args = parser.parse_args()

    apply_theme()
    warnings.filterwarnings("ignore", category=UserWarning)
    args.out.mkdir(parents=True, exist_ok=True)

    # XFOIL is the default solver — `--synthetic` flips to the closed-form model.
    use_xfoil = not args.synthetic

    solver_label = f"real XFOIL ({args.xfoil_binary})" if use_xfoil else "closed-form synthetic"
    print(f"Output root: {args.out.resolve()}")
    print(f"Solver: {solver_label}")
    print(f"Mission profile ({len(MISSION)} points):")
    for mp in MISSION:
        print(
            f"  • {mp.pretty}: α = {mp.alpha}°, Re = {mp.reynolds:.0e}, "
            f"M = {mp.mach:.2f}, weight = {mp.lift_weight}"
        )

    runs: list[RunResult] = []

    # ---------- STAGE 1: mono-point per mission ----------
    if not args.skip_stage1:
        print("\n=== STAGE 1 — Mono-point optimisations ===")
        for mp in MISSION:
            single = (mp,)
            solver = make_solver(use_xfoil, args, single, primary_name=mp.name)
            run = run_one_optimisation(
                name=f"stage1_{mp.name}",
                pretty=f"Stage 1 — {mp.pretty} only",
                solver=solver,
                mission=single,
                primary_name=mp.name,
                pop=args.pop,
                n_gen=args.n_gen,
                seed=args.seed,
            )
            runs.append(run)
            render_full_asset_set(
                run,
                args.out / run.name,
                fps=args.fps,
                fade_window=args.fade_window,
                f1_label=axis_label("cd") + f" — {mp.pretty}",
                f2_label=axis_label("cl") + f" — {mp.pretty}",
            )

    # ---------- STAGE 2: multi-point ----------
    if not args.skip_stage2:
        print("\n=== STAGE 2 — Multi-point optimisation ===")
        solver = make_solver(use_xfoil, args, MISSION, primary_name="cruise")
        run = run_one_optimisation(
            name="stage2_multipoint",
            pretty="Stage 2 — Multi-point (whole mission)",
            solver=solver,
            mission=MISSION,
            primary_name="cruise",
            pop=args.pop,
            n_gen=args.n_gen,
            seed=args.seed,
        )
        runs.append(run)
        render_full_asset_set(
            run,
            args.out / run.name,
            fps=args.fps,
            fade_window=args.fade_window,
            f1_label=axis_label("cd_cr"),
            f2_label=axis_label("cl_tot"),
        )

    # ---------- STAGE 3: cross-comparison ----------
    if len(runs) >= 2:
        print("\n=== STAGE 3 — Cross-comparison ===")
        cmp_dir = args.out / "comparison"
        cmp_dir.mkdir(parents=True, exist_ok=True)
        render_comparison_champion_geometries(runs, cmp_dir / "champion_geometries.png")
        render_comparison_mission_performance(runs, cmp_dir / "champion_mission_performance.png")
        render_comparison_design_parameters(runs, cmp_dir / "champion_design_parameters.png")
        render_comparison_pareto_overlay(runs, cmp_dir / "pareto_overlay.png")
        print(f"  comparison assets → {cmp_dir.resolve()}")

    print("\nDone. Drop the assets onto your portfolio site.")


if __name__ == "__main__":
    main()
