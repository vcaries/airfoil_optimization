# Using aeroforge

A practical, layer-by-layer walkthrough of the library. Each section is
self-contained and uses only the imports it actually needs. Code snippets
are copy-pasteable — paste them into `python -i` or a Jupyter cell.

> If you only have 30 seconds: run `python examples/04_full_pipeline.py`
> and inspect the PNG + GIF gallery it drops into `docs/assets/`. That
> single script exercises every layer described below. By default it
> drives the real XFOIL binary; add `--synthetic` if XFOIL is not on
> your `PATH`.

---

## Table of contents

1. [Installation](#1-installation)
2. [Geometry: generate, inspect, transform, save](#2-geometry-generate-inspect-transform-save)
3. [Running a polar through XFOIL](#3-running-a-polar-through-xfoil)
4. [Convergence strategies for stubborn points](#4-convergence-strategies-for-stubborn-points)
5. [Optimization with pymoo](#5-optimization-with-pymoo)
6. [Campaigns with resumable persistence](#6-campaigns-with-resumable-persistence)
7. [Visualization and portfolio animations](#7-visualization-and-portfolio-animations)
8. [Extending: add your own generator / strategy / objective](#8-extending-add-your-own-generator--strategy--objective)
9. [CLI cheat sheet](#9-cli-cheat-sheet)

---

## 1. Installation

```bash
# Minimal core (geometry + solver interface)
pip install -e .

# Everything (optimization, plotting, IO, CLI)
pip install -e ".[all]"

# Developer setup: tests, ruff, mypy, mkdocs, pre-commit
make dev
```

**XFOIL binary.** The solver layer and `examples/04_full_pipeline.py`
use it by default. Install it separately and make sure `xfoil` is on
your `PATH` (or pass an explicit path to `XfoilRunner(binary=...)` /
`--xfoil-binary`). Everything else — geometry, optimization, animation
— works without XFOIL, and example 04 also exposes `--synthetic` for
that case.

To check what you have:

```python
import shutil, aeroforge
print("aeroforge:", aeroforge.__version__)
print("xfoil:", shutil.which("xfoil") or "NOT installed")
```

---

## 2. Geometry: generate, inspect, transform, save

```python
from aeroforge.geometry import NACA4Generator

airfoil = NACA4Generator(designation="2412", n_points=120).generate()
print(airfoil)
# Airfoil(name='NACA 2412', n_points=239, max_thickness=0.1200, max_camber=0.0200)

# Derived metrics (all properties, no recomputation needed)
airfoil.max_thickness            # 0.1199...
airfoil.max_thickness_location   # 0.30  (x/c)
airfoil.max_camber               # 0.0200
airfoil.max_camber_location      # 0.40  (x/c)
airfoil.area                     # enclosed area, ~0.082
airfoil.trailing_edge_gap        # 0.0 for closed-TE NACA 4-digit

# Upper / lower surfaces (LE -> TE, monotonic x)
xu, yu, xl, yl = airfoil.surfaces()

# Affine transforms — they return new Airfoils, never mutate the source.
rotated = airfoil.rotated(5.0)             # 5 deg counter-clockwise
moved   = airfoil.translated(0.25, -0.05)
scaled  = airfoil.scaled(0.5)
unit    = scaled.normalized()              # back to a canonical unit chord

# I/O — Selig-format .dat file (the format XFOIL reads).
airfoil.to_dat("naca2412.dat")
from aeroforge.geometry import Airfoil
reloaded = Airfoil.from_dat("naca2412.dat")
```

### Loading an existing airfoil

```python
from aeroforge.geometry import DatFileGenerator

# Optionally repanel to a target node count for XFOIL.
e387 = DatFileGenerator("airfoils/e387.dat", repanel_to=160).generate()
```

### Open vs closed trailing edge

```python
from aeroforge.core.types import TrailingEdge
NACA4Generator("0012", trailing_edge=TrailingEdge.OPEN).generate().trailing_edge_gap
# ~2e-3 — the classic finite-TE polynomial
NACA4Generator("0012", trailing_edge=TrailingEdge.CLOSED).generate().trailing_edge_gap
# 0.0 — modified last coefficient so the surfaces meet exactly
```

---

## 3. Running a polar through XFOIL

```python
from aeroforge.core.types import OperatingPoint
from aeroforge.geometry import NACA4Generator
from aeroforge.solver import XfoilRunner

airfoil = NACA4Generator("2412", n_points=120).generate()
runner = XfoilRunner(binary="xfoil", timeout_s=30.0, max_iter=200)

result = runner.analyze(airfoil, OperatingPoint(alpha=4.0, reynolds=1.0e6))
print(result.cl, result.cd, result.lift_to_drag)
```

`XfoilRunner` owns the subprocess lifecycle: it writes a scratch `.dat`,
renders the stdin command transcript, invokes `xfoil` with a hard timeout,
parses the polar dump, and returns a typed `PolarPoint`. Process-level
failures become typed exceptions:

- `XfoilNotFoundError` — binary not on PATH (raised at construction).
- `XfoilExecutionError` — non-zero exit, hang/timeout, OS-level failure.
- `ConvergenceError` — XFOIL ran but the requested α never converged.

### Sweeping a polar

```python
import numpy as np
from aeroforge.solver.xfoil.results import Polar

polar = Polar()
for alpha in np.arange(-2.0, 12.5, 0.5):
    try:
        polar.points.append(
            runner.analyze(airfoil, OperatingPoint(alpha=float(alpha), reynolds=1e6))
        )
    except Exception:
        polar.failed.append(OperatingPoint(alpha=float(alpha), reynolds=1e6))

print(f"{len(polar)} converged, {len(polar.failed)} failed")
```

(See [`examples/02_run_polar.py`](examples/02_run_polar.py) for the same
loop with a synthetic-solver fallback for when XFOIL is not installed.)

---

## 4. Convergence strategies for stubborn points

XFOIL is sometimes touchy near stall, with low Re, or on airfoils with
awkward panel spacing. Wire one or more strategies behind a
`ConvergencePipeline` and they will be tried in order whenever the solver
raises `ConvergenceError`:

```python
from aeroforge.solver.convergence import (
    AlphaContinuationStrategy,
    ConvergencePipeline,
    IncreaseIterationsStrategy,
    PerturbAlphaStrategy,
    RepanelStrategy,
)

pipeline = ConvergencePipeline([
    IncreaseIterationsStrategy(factor=2.0, max_iter=800),
    RepanelStrategy(),
    AlphaContinuationStrategy(step=0.25, max_steps=20),
    PerturbAlphaStrategy(epsilon=0.05),
])

# Pass `pipeline` into a CampaignRunner (next section) or call it directly:
history = []   # converged neighbours; matters for AlphaContinuation
try:
    pt = runner.analyze(airfoil, op)
except ConvergenceError:
    pt = pipeline.attempt(runner, airfoil, op, history=history)
history.append(pt)
```

Strategies always restore the solver's mutated attributes (`max_iter`,
`repanel`, ...) in a `finally` block, so they compose safely in any order.

---

## 5. Optimization with pymoo

Three steps: define a design space, plug in objectives + constraints,
launch a study.

```python
from aeroforge.core.types import OperatingPoint
from aeroforge.geometry import Airfoil, NACA4Generator
from aeroforge.optimization import (
    AirfoilEvaluator,
    DesignSpace,
    DesignVariable,
    MaximizeLiftToDrag,
    MinThicknessConstraint,
    OptimizationStudy,
)
from aeroforge.optimization.algorithms import ga
from aeroforge.solver import XfoilRunner


# (a) the design space — three continuous variables decoded into a NACA 4-digit
def build_naca4(params: dict[str, float]) -> Airfoil:
    m = int(round(params["m"] * 9))
    p_digit = max(int(round(params["p"] * 9)), 1)
    t_pct = max(int(round(params["t"] * 24)) + 6, 6)
    return NACA4Generator(f"{m}{p_digit}{t_pct:02d}", n_points=120).generate()

space = DesignSpace([
    DesignVariable("m", 0.0, 0.9),
    DesignVariable("p", 0.1, 1.0),
    DesignVariable("t", 0.0, 1.0),
])

# (b) the evaluator — bridge between pymoo and aeroforge
evaluator = AirfoilEvaluator(
    design_space=space,
    airfoil_factory=build_naca4,
    solver=XfoilRunner("xfoil"),
    operating_point=OperatingPoint(alpha=4.0, reynolds=5e5),
    objectives=[MaximizeLiftToDrag()],
    geometric_constraints=[MinThicknessConstraint(t_min=0.08)],
)

# (c) the study — pick an algorithm, set generations + seed
study = OptimizationStudy(
    evaluator=evaluator,
    algorithm=ga(pop_size=40),
    n_gen=50,
    seed=42,
)
result = study.run()

best = evaluator.genome_to_airfoil(result.X)
print(f"Best: {best.name}  L/D = {-float(result.F[0]):.2f}")
```

### Notes

- **Failures are soft.** If the airfoil factory or solver raises, the
  evaluator returns sentinel values (large F, large G) so pymoo dominates
  the bad candidate without crashing.
- **Geometric prune first.** Geometric constraints are evaluated before
  the solver call. A candidate that's already infeasible never costs you
  an XFOIL invocation.
- **Reproducibility.** Same seed + same evaluator + same algorithm =
  bit-identical history. This is what makes the GIFs reproducible.

### Multi-objective

Swap `ga` for `nsga2` and add a second objective:

```python
from aeroforge.optimization import MinimizeDrag
from aeroforge.optimization.algorithms import nsga2

evaluator = AirfoilEvaluator(
    design_space=space,
    airfoil_factory=build_naca4,
    solver=XfoilRunner("xfoil"),
    operating_point=OperatingPoint(alpha=4.0, reynolds=5e5),
    objectives=[MaximizeLiftToDrag(), MinimizeDrag()],
)
study = OptimizationStudy(
    evaluator=evaluator,
    algorithm=nsga2(pop_size=60),
    n_gen=80,
    seed=42,
)
result = study.run()
# result.X has the Pareto-optimal genomes, result.F their (-L/D, C_d) pairs.
```

### Checkpoint + resume

```python
study = OptimizationStudy(
    evaluator=evaluator,
    algorithm=ga(pop_size=40),
    n_gen=50,
    seed=42,
    checkpoint_path="runs/ld_study.pkl",
)
study.run()           # saves history at the end

# Later, in a new process:
study.resume()        # loads history, runs again from the deterministic seed
```

(See [`examples/03_optimize_ld.py`](examples/03_optimize_ld.py) for a
fully runnable single-objective example with a synthetic solver.)

---

## 6. Campaigns with resumable persistence

Same idea as an optimization but for one-off parameter sweeps. Useful for
exploring a Re/α grid, or for pre-computing the seed population of an
optimizer.

```python
from aeroforge.campaigns import CampaignRunner, Sweep, SqliteResultStore
from aeroforge.core.types import OperatingPoint
from aeroforge.geometry import NACA4Generator
from aeroforge.solver import XfoilRunner

airfoil = NACA4Generator("2412").generate()
sweep = Sweep(
    parameters={
        "alpha": [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0],
        "re":    [3e5, 1e6, 3e6],
    },
    factory=lambda p: (
        airfoil,
        OperatingPoint(alpha=p["alpha"], reynolds=p["re"]),
    ),
)

with SqliteResultStore("runs/sweep.db") as store:
    result = CampaignRunner(
        solver=XfoilRunner("xfoil"),
        store=store,
        n_workers=4,   # process-level parallelism
    ).run(sweep)

print(f"{len(result.converged)}/{len(sweep)} converged "
      f"({result.success_rate:.0%})")
```

**Resume by re-running.** Kill the script halfway, relaunch with the same
`SqliteResultStore` — every cached `(airfoil_hash, OperatingPoint)` is
skipped, the solver is only invoked for the remaining points.

> Parallel mode caveat: convergence strategies that consult the live
> history (e.g. `AlphaContinuationStrategy`) only see an empty history in
> worker processes. Stick to serial mode (`n_workers=1`) if alpha
> continuation is critical for your case.

---

## 7. Visualization and portfolio animations

### Static plots

```python
import matplotlib.pyplot as plt
from aeroforge.visualization import (
    plot_geometry, plot_polar, plot_cp,
    use_portfolio_style,
)

use_portfolio_style()                    # consistent visual identity
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plot_geometry(airfoil, ax=axes[0])
plot_cp(cp_distribution, ax=axes[1])
fig.savefig("snapshot.png", dpi=150, bbox_inches="tight")
```

Every plot function:

- accepts an optional `ax` so you can compose multi-panel figures,
- returns the `Axes` it drew on,
- never calls `plt.show()`, so it works headless / on CI.

### Pareto front

```python
from aeroforge.visualization import plot_pareto_front
last_snap = study.history.snapshots[-1]
ax = plot_pareto_front(last_snap)
```

### Geometry evolution GIF

```python
from aeroforge.visualization import animate_geometry_evolution

animate_geometry_evolution(
    history=study.history,
    genome_to_airfoil=evaluator.genome_to_airfoil,
    output="docs/assets/geometry.gif",
    fps=4,
    show_baseline=True,   # overlay generation 0 in faded outline
)
```

Output format is auto-detected from the extension: `.gif` (imageio) or
`.mp4` (imageio + ffmpeg). The animation locks the axis bounds across
frames so the shape change is the only visible motion.

### Pareto-front animation

```python
from aeroforge.visualization import animate_pareto_evolution
animate_pareto_evolution(study.history, "docs/assets/pareto.gif", fps=6)
```

---

## 8. Extending: add your own generator / strategy / objective

The architecture is built around four small ABCs. Adding a new
implementation is a one-file change.

### A new airfoil parameterization

```python
from aeroforge.geometry import Airfoil
from aeroforge.geometry.generators.base import AirfoilGenerator

class CircularArcGenerator(AirfoilGenerator):
    def __init__(self, camber: float, thickness: float, n_points: int = 80):
        self.camber, self.thickness, self.n_points = camber, thickness, n_points

    @property
    def name(self) -> str:
        return f"CircArc(c={self.camber}, t={self.thickness})"

    def generate(self) -> Airfoil:
        ...   # build x/y arrays in Selig order
        return Airfoil(x, y, name=self.name)
```

That's it — every layer that consumes an `AirfoilGenerator` (campaigns,
optimization) accepts it unchanged.

### A new convergence strategy

```python
from aeroforge.solver.convergence.base import ConvergenceStrategy

class HalfStepStrategy(ConvergenceStrategy):
    def attempt(self, solver, airfoil, point, *, history):
        ...   # mutate solver attrs, call solver.analyze, return / raise
```

Drop it into a `ConvergencePipeline` and you're done.

### A new objective

```python
from aeroforge.optimization.objectives import Objective

class MinimizeMoment(Objective):
    def evaluate(self, result) -> float:
        return abs(float(result.cm))
```

Add it to the evaluator's `objectives=[...]`. `AirfoilProblem.n_obj`
picks it up automatically.

---

## 9. CLI cheat sheet

Install the CLI extra (`pip install -e ".[cli]"` or `[all]`) and use:

```bash
# Generate a NACA 4-digit airfoil and save it
aeroforge airfoil naca 2412 --n 120 --out naca2412.dat

# Show the version
aeroforge --version
```

More subcommands (polar runner, optimization launcher) are coming in v0.2 —
the CLI surface is intentionally minimal in v0.1 to keep the focus on the
Python API.

---

## See also

- [`README.md`](README.md) — high-level overview + badges.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — design rationale, layered module
  map, design patterns, roadmap, Git workflow.
- [`CHANGELOG.md`](CHANGELOG.md) — what changed in each milestone.
- [`examples/`](examples/) — runnable end-to-end demos.
