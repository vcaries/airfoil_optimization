# aeroforge

Welcome to the API documentation for **aeroforge**, a professional
object-oriented Python toolkit for 2D airfoil aerodynamics built around the
XFOIL solver.

Start with the [project README](https://github.com/vcaries/airfoil_optimization)
for installation and a quickstart, and the
[architecture document](https://github.com/vcaries/airfoil_optimization/blob/main/ARCHITECTURE.md)
for the design rationale.

## Package layout

- `aeroforge.core` — exceptions, logging, shared types.
- `aeroforge.geometry` — airfoil representation and generation.
- `aeroforge.solver` — the XFOIL wrapper and convergence strategies.
- `aeroforge.campaigns` — parametric sweeps and batch execution.
- `aeroforge.optimization` — pymoo-based single/multi-objective optimization.
- `aeroforge.visualization` — plots and animations.

::: aeroforge.geometry
