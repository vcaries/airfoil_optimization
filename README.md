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

> **Status:** active development. The geometry engine (NACA 4-digit) is
> implemented and tested; the solver, optimization, and visualization layers
> are scaffolded against stable public interfaces. See
> [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design and roadmap.

---

## Why this project

`aeroforge` turns [XFOIL](https://web.mit.edu/drela/Public/web/xfoil/) — the
de-facto 2D viscous airfoil solver — into a robust, scriptable, and extensible
scientific library. It is built to demonstrate engineering-grade scientific
software: clean object-oriented design, defensive handling of solver
non-convergence, reproducible computational campaigns, and automated generation
of the figures and animations used to communicate results.

## Feature map

| Layer | What it does | State |
|-------|--------------|-------|
| **Geometry** | NACA 4-digit, parametric (CST/Bézier/PARSEC), `.dat` import; repaneling, smoothing, transforms; geometric metrics | NACA 4-digit ✅ · others scaffolded |
| **Solver** | Automated XFOIL process control, command DSL, output parsing (polars, Cp, BL), cross-platform | Scaffolded |
| **Convergence** | Pluggable strategies: iteration ramp-up, alpha continuation, smart restart, stall handling, adaptive retries | Scaffolded |
| **Campaigns** | Parametric sweeps, parallel batch execution, result persistence, resumable runs | Scaffolded |
| **Optimization** | Generic [pymoo](https://pymoo.org/) problem: any design variables, geometric/physical constraints, single- & multi-objective | Scaffolded |
| **Visualization** | Polars, Cp distributions, Pareto fronts, convergence history, and GIF/MP4 animations of design evolution | Scaffolded |

## Installation

```bash
# Core (geometry + solver interface)
pip install -e .

# Everything (optimization, plotting, IO, CLI)
pip install -e ".[all]"

# Developer setup (tests, linting, typing, docs, pre-commit)
make dev
```

### XFOIL binary

The solver layer shells out to the `xfoil` executable, which must be installed
separately and reachable on your `PATH` (or pointed to via configuration):

- **Linux/macOS**: build from the [XFOIL source](https://web.mit.edu/drela/Public/web/xfoil/)
  or install via your package manager where available.
- **Windows**: download the official binary and add its folder to `PATH`.

The geometry engine has **no** XFOIL dependency and works standalone.

## Quickstart

```python
from aeroforge.geometry import NACA4Generator

# Generate a NACA 2412 with 160 cosine-spaced panel nodes.
airfoil = NACA4Generator(designation="2412", n_points=160).generate()

print(airfoil)                       # Airfoil(name='NACA 2412', n_points=...)
print(airfoil.max_thickness)         # ~0.12  (12% chord)
print(airfoil.max_camber)            # ~0.02  (2% chord)

airfoil.to_dat("naca2412.dat")       # Selig-format coordinate file for XFOIL
```

See [`examples/`](examples/) for end-to-end scripts (polar sweeps, L/D
optimization, animation rendering).

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — design rationale, module map, patterns,
  roadmap, and testing/visualization/Git strategies.
- API reference (mkdocs-material + mkdocstrings) is published to GitHub Pages.

## License

[MIT](LICENSE) © Valentin Caries · Part of my freelance scientific-computing
portfolio at [vcaries.github.io](https://vcaries.github.io).
