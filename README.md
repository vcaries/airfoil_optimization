<div align="center">

# aeroforge

**A professional, object-oriented Python toolkit for 2D airfoil aerodynamics:
an advanced XFOIL wrapper, geometry engine, multi-objective optimizer, and
publication-grade visualization pipeline.**

[![CI](https://github.com/vcaries/airfoil_optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/vcaries/airfoil_optimization/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

> **Status:** v0.2 — all five layers (geometry, solver, optimization,
> campaigns, visualization) are implemented and unit-tested, and the
> end-to-end portfolio demo drives the real XFOIL binary by default. See
> [`ARCHITECTURE.md`](ARCHITECTURE.md) for the design and
> [`USAGE.md`](USAGE.md) for a step-by-step tutorial.

---

## Why this project

`aeroforge` turns [XFOIL](https://web.mit.edu/drela/Public/web/xfoil/) — the
de-facto 2D viscous airfoil solver — into a robust, scriptable, and extensible
scientific library. Five clean layers (geometry, solver, campaigns,
optimization, visualization) plug together to take you from a NACA designation
to a Pareto front animation in under 100 lines of Python.

## Feature map

| Layer | What it does | State |
|-------|--------------|-------|
| **Geometry** | NACA 4-digit, `.dat` import, repaneling, smoothing, transforms; geometric metrics | ✅ Implemented |
| **Solver** | Automated XFOIL process control, command DSL, polar + Cp parsing, cross-platform | ✅ Implemented |
| **Convergence** | Pluggable strategies: iteration ramp-up, alpha continuation, repanel, perturb-alpha, inviscid init | ✅ Implemented |
| **Campaigns** | Parametric sweeps, parallel batch execution, resumable SQLite/Parquet stores | ✅ Implemented |
| **Optimization** | Generic [pymoo](https://pymoo.org/) problem: any design variables, geometric/physical constraints, single- & multi-objective | ✅ Implemented |
| **Visualization** | Polars, Cp distributions, Pareto fronts, convergence history, GIF/MP4 animations of design evolution | ✅ Implemented |

Parametric shape families (CST / Bézier / PARSEC) are scaffolded behind their
interfaces; full implementations land in v0.2 (see roadmap in
[`ARCHITECTURE.md`](ARCHITECTURE.md)).

## Installation

```bash
# Core (geometry + solver interface)
pip install -e .

# Everything (optimization, plotting, IO, CLI)
pip install -e ".[all]"

# Developer setup (tests, linting, typing, docs, pre-commit)
make dev
```

### XFOIL binary (optional but recommended)

The solver layer shells out to the `xfoil` executable, which must be installed
separately and reachable on your `PATH` (or pointed to via configuration):

- **Linux/macOS**: build from the [XFOIL source](https://web.mit.edu/drela/Public/web/xfoil/)
  or install via your package manager where available.
- **Windows**: download the official binary and add its folder to `PATH`.

The geometry engine, the optimization layer, the visualization pipeline, and
example scripts 01–03 work without XFOIL. The end-to-end portfolio demo
[`examples/04_full_pipeline.py`](examples/04_full_pipeline.py) drives the
real XFOIL binary by default; pass `--synthetic` to use the closed-form
fallback when XFOIL is not available.

## Quickstart

The end-to-end demo:

```bash
# Real XFOIL (default) — runs the full mission-aware multi-objective workflow
python examples/04_full_pipeline.py --out docs/assets

# Closed-form fallback (no XFOIL needed)
python examples/04_full_pipeline.py --synthetic --out docs/assets
```

Runs N mono-point NSGA-II optimisations (one per mission leg) plus one
multi-point optimisation, then renders a full PNG + GIF gallery per run
plus cross-comparison figures into `docs/assets/`:

- `stage1_<leg>/pareto_evolution.gif` — objective space generation by generation
- `stage1_<leg>/design_evolution.gif` — design-space projections
- `stage1_<leg>/parallel_evolution.gif` — parallel coordinates of the population
- `stage1_<leg>/geometry_evolution.gif` — Pareto geometries evolving
- `stage1_<leg>/{convergence,pareto_geometries,mission_breakdown,multipoint_polars}.png`
- `stage2_multipoint/…` — same gallery for the multi-point run
- `comparison/{champion_geometries,champion_mission_performance,champion_design_parameters,pareto_overlay}.png`

A 10-line teaser:

```python
from aeroforge.geometry import NACA4Generator

airfoil = NACA4Generator(designation="2412", n_points=120).generate()
print(airfoil)                       # Airfoil(name='NACA 2412', n_points=239, ...)
print(airfoil.max_thickness)         # ~0.12
print(airfoil.max_camber)            # ~0.02
airfoil.to_dat("naca2412.dat")       # Selig-format coordinate file for XFOIL
```

Other runnable examples:

- [`examples/01_generate_airfoil.py`](examples/01_generate_airfoil.py) — NACA 4-digit + metrics
- [`examples/02_run_polar.py`](examples/02_run_polar.py) — α sweep via XFOIL (or synthetic fallback)
- [`examples/03_optimize_ld.py`](examples/03_optimize_ld.py) — GA optimising L/D
- [`examples/04_full_pipeline.py`](examples/04_full_pipeline.py) — end-to-end portfolio demo

For a step-by-step walkthrough of every layer, read [`USAGE.md`](USAGE.md).

## Documentation

- [`USAGE.md`](USAGE.md) — user-facing tutorial covering every layer.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — design rationale, module map, design
  patterns, roadmap, and Git workflow.
- API reference (mkdocs-material + mkdocstrings) is published to GitHub Pages.

## License

[MIT](LICENSE) © Valentin Caries · Part of my freelance scientific-computing
portfolio at [vcaries.github.io](https://vcaries.github.io).
