# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] — 2026-05

### Added
- Surface-distribution gallery in `examples/04_full_pipeline.py`:
  per-run GIFs and final PNGs for **Cp, Cf, H, δ*, θ, and Ue/Vinf** at
  every mission point of every stage, plus stage-3 cross-comparison
  figures overlaying each champion's distribution at each leg. The
  renderer factory is descriptor-driven (`WallQuantity` dataclass +
  `WALL_QUANTITIES` tuple), so adding or removing a quantity is a
  one-line change.
- New library entry point
  `XfoilRunner.analyze_with_dumps(airfoil, point)` returns
  `(PolarPoint, CpDistribution | None, WallProfile | None)`. The
  existing `analyze` API is unchanged.
- `XfoilCommand.dump_bl(path)` emits the OPER `DUMP` command for
  boundary-layer profiles.
- `XfoilOutputParser.parse_bl_dump(path, point)` parses the standard
  XFOIL DUMP file (`s x y Ue/Vinf Dstar Theta Cf H`).
- `WallProfile` dataclass exposing all eight DUMP columns, panel-order.
- `XfoilSession.to_command_script` now accepts optional `cp_path` and
  `bl_path` kwargs to opt into surface dumps.
- Best-effort post-mortem diagnostics on `XfoilRunner`:
  `last_transcript`, `last_stdout`, `last_stderr`, `last_workdir_files`,
  and `last_files` are populated after every analyze call so callers
  can inspect what XFOIL actually wrote without rerunning.

### Fixed
- `XfoilOutputParser.parse_cp` is now tolerant of both 2-column
  (`x Cp`) and 3-column (`x y Cp`) CPWR output variants, and silently
  skips non-numeric lines instead of raising. Some XFOIL builds emit
  header rows that defeated the strict parser.
- `XfoilRunner._run_in` falls back to XFOIL's default dump filenames
  (`<basename>.cp` / `<basename>.bl`) when the explicitly requested
  filename does not materialise, so the Cp / BL dumps are picked up
  regardless of the build's filename quirks.
- `XfoilSession.to_command_script` emits `CPWR` + `DUMP` **before**
  closing the polar accumulator. Some XFOIL builds short-circuit
  those commands when issued after a `PACC` close.

## [0.2.1] — 2026-05

### Fixed
- Make XFOIL actually usable on Windows:
  - `XfoilRunner.analyze` now feeds the stdin transcript as raw bytes
    (`text=False`) so Python no longer translates `\n` into `\r\n` — some
    XFOIL builds silently dropped any command followed by `\r`.
  - DAT and PACC filenames are passed as basenames (the subprocess already
    runs with `cwd=workdir`). XFOIL's FORTRAN parser truncates filenames
    at the first `:`, so the previous absolute paths
    (`C:\Users\…\airfoil.dat`) were read as just `C` and every LOAD
    failed with "Nonexistent file: C".
  - The XFOIL console window is suppressed via `CREATE_NO_WINDOW` plus a
    `STARTUPINFO` with `STARTF_USESHOWWINDOW` + `SW_HIDE`, so long GA
    runs no longer flicker thousands of cmd windows on Windows.
- `ConvergenceError` now embeds the XFOIL transcript and stdout/stderr
  tails so future failures are diagnosable in one log line.

### Added — Robustness in `examples/04_full_pipeline.py`
- `AggregatedXfoilSolver` translates per-point XFOIL failures to
  `SolverError` (caught by `AirfoilEvaluator.evaluate` and turned into
  sentinel values) instead of raising `RuntimeError`, which used to
  abort the whole pymoo run.
- Per-run feasibility diagnostic prints how many candidates were
  geometrically infeasible, XFOIL-failed, physically-infeasible, or
  fully feasible, plus best CL/CD seen — empty-Pareto symptoms are now
  diagnosable in one glance.
- First-N XFOIL exception traces (default 3, configurable via
  `debug_max_traces`) so initial failures of a long run are shown
  verbatim instead of swallowed.

## [0.2.0] — 2026-05

### Added — M6: Portfolio polish & mission-aware demo
- `examples/04_full_pipeline.py` rewritten as a three-stage,
  mission-aware showcase (N mono-point NSGA-II runs + one multi-point
  run + cross-comparison gallery). Fully scalable in the number of
  mission points; every figure, GIF, and legend adapts automatically.
- Mission-differentiating physical model with Reynolds-, Mach- and
  stall-aware lift/drag laws plus a full mechanical / manufacturing
  geometric constraint stack (`MinEnclosedAreaConstraint`,
  `MinThicknessAtConstraint`, `MaxTEGapConstraint`,
  `MaxAbsCamberConstraint`).
- Engineering aerodynamic envelope, applied as `PhysicalConstraint`s
  on the aggregated PolarPoint: `MinLiftConstraint`,
  `MaxDragConstraint`, `MaxAbsPitchingMomentConstraint`. Thresholds
  are looked up per mission point via `ENGINEERING_LIMITS`, so each
  stage-1 mono-point run optimises under its own (CL floor, CD ceiling,
  trim-authority budget) — driving the take-off, cruise, and landing
  optima toward genuinely different airfoils.
- Cumulative accumulator + standardised population legends across every
  evolution plot; Computer Modern serif theme with unicode-minus
  disabled for clean LaTeX-like typography on every platform.
- Convergence panel extended with hypervolume history (pymoo `HV`
  indicator) alongside best-lift, best-drag, and Pareto-set-size.
- Knee-point (closest-to-utopia) recommendation marker highlighted in
  every objective-, design-, parallel-, and geometry-evolution figure.

### Changed
- `examples/04_full_pipeline.py` now drives the **real XFOIL binary by
  default**. The previous `--xfoil` opt-in is replaced by an explicit
  `--synthetic` opt-out, and the script aborts with a clear message if
  XFOIL is requested but not on `PATH`.
- README and USAGE updated to reflect the XFOIL-by-default behaviour
  and the new asset gallery layout under `docs/assets/`.

### Added — M5: Visualization & portfolio assets
- `aeroforge.visualization.plots`: `plot_geometry`, `plot_cl_alpha`,
  `plot_drag_polar`, `plot_polar` (two-panel composer), `plot_cp` (with
  inverted y-axis), `plot_convergence_history`.
- `aeroforge.visualization.pareto`: `plot_pareto_front` plus a public
  `non_dominated_mask` helper.
- `aeroforge.visualization.animation`: `animate_geometry_evolution` (best
  airfoil per generation with optional faded baseline) and
  `animate_pareto_evolution`. GIF (imageio) and MP4 (imageio-ffmpeg)
  auto-detected by extension; per-frame `Figure` close keeps memory bounded.
- `examples/04_full_pipeline.py`: end-to-end demo emitting `polar.png`,
  `geometry.gif`, `convergence.png`, `final_airfoil.png`.
- 16 unit tests for the visualization layer (matplotlib Agg backend).

### Added — M4: Campaigns layer
- `aeroforge.campaigns.store`: `SqliteResultStore` (idempotent on
  `(airfoil_hash, alpha, Re, Mach, n_crit)`) and `ParquetResultStore`
  (in-memory + flush) sharing the `ResultStore` ABC.
- `hash_airfoil` helper (SHA-1 over coordinates, 16 hex chars).
- `aeroforge.campaigns.runner`: `CampaignRunner` with serial and
  `multiprocessing.Pool`-backed parallel execution paths; the parent owns
  the store to avoid pickling SQLite connections. Resumability proven by
  test: re-run on the same store skips every cached point.
- `CampaignResult` value object with `success_rate`.
- 17 unit tests for the campaigns layer.

### Added — M3: Optimization layer (pymoo)
- `aeroforge.optimization.evaluator.AirfoilEvaluator.evaluate`: full
  pipeline — genome decode -> airfoil factory -> geometric prune ->
  solver -> physical constraints -> objectives. Failures translated to
  finite sentinel values so pymoo never crashes on a bad candidate.
- `aeroforge.optimization.problem.AirfoilProblem._evaluate`: pymoo
  `Problem` adapter with constraint-aware population evaluation.
- `aeroforge.optimization.callbacks.HistoryCallback`: properly subclasses
  pymoo's `Callback`, exposes `best_per_generation()` and `__len__`.
- `aeroforge.optimization.study.OptimizationStudy`: high-level facade with
  `run`, `save_checkpoint`, `load_checkpoint`, `resume`.
- `AirfoilEvaluator.genome_to_airfoil` convenience helper for the
  visualization layer.
- 12 unit tests; end-to-end convergence demonstrated against a synthetic
  L/D peak.

### Added — M2: Working XFOIL wrapper
- `aeroforge.solver.xfoil.parser.XfoilOutputParser`: polar (PACC) and Cp
  (CPWR) parsers with header extraction and typed `ParsingError` on
  malformed input.
- `aeroforge.solver.xfoil.runner.XfoilRunner.analyze`: scratch tempdir,
  stdin command transcript, subprocess invocation with hard timeout,
  typed exception translation (`XfoilExecutionError`, `XfoilNotFoundError`,
  `ConvergenceError`).
- `aeroforge.solver.convergence.strategies`: five concrete strategies
  (`IncreaseIterationsStrategy`, `AlphaContinuationStrategy`,
  `PerturbAlphaStrategy`, `RepanelStrategy`, `InviscidInitStrategy`).
  Each restores any mutated solver attributes in a `finally` block.
- Canonical XFOIL output fixtures under `tests/data/`.
- 34 unit tests across parser / runner (subprocess mocked) / convergence
  (FakeSolver driven).

### Changed
- `.gitignore`: added negation rules for `tests/data/**/*.pol|cp|bl` so the
  fixture files aren't swept up by the XFOIL scratch-file ignores.
- mypy `ignore_missing_imports` extended for `pandas`, `pyarrow`,
  `pydantic_settings`, `cycler`, `typer`, `rich`, `imageio_ffmpeg`.
- `.pre-commit-config.yaml`: mypy hook now installs `pydantic-settings`.

### Tests
**150 unit tests, all green.** Coverage of implemented modules: `Airfoil`
95%, `core.types` 96%, `core.exceptions` 88%, geometry operations 76%+.

## [0.1.0-M1] — 2026-05

### Added — M1: Geometry foundation
- Repository scaffolding (src layout, pyproject, CI matrix, pre-commit,
  mkdocs site).
- `aeroforge.core`: exception hierarchy, structured logging, shared types
  (`FloatArray`, `OperatingPoint`, `Surface`, `TrailingEdge`).
- `aeroforge.geometry`: `Airfoil` value object, `AirfoilGenerator` ABC,
  `NACA4Generator` (fully tested), `DatFileGenerator`, cosine spacing,
  repanel, affine transforms, geometric metrics.
- Stub interfaces (with `NotImplementedError` bodies) for solver,
  convergence, campaigns, optimization, visualization, CLI.
- `ARCHITECTURE.md` (~700 lines) covering goals, layered architecture,
  repo layout, module map, design patterns, public ABCs, roadmap,
  dependencies, testing/visualization/Git strategies.

[Unreleased]: https://github.com/vcaries/airfoil_optimization/compare/v0.3.0...main
[0.3.0]: https://github.com/vcaries/airfoil_optimization/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/vcaries/airfoil_optimization/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/vcaries/airfoil_optimization/compare/v0.1.0-M1...v0.2.0
[0.1.0-M1]: https://github.com/vcaries/airfoil_optimization/releases/tag/v0.1.0-M1
