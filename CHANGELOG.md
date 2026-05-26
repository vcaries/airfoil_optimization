# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project architecture and repository scaffolding (`ARCHITECTURE.md`).
- `aeroforge.core`: exception hierarchy, structured logging, shared types.
- `aeroforge.geometry`: `Airfoil` value object, generator abstraction, and a
  fully implemented, unit-tested `NACA4Generator`.
- `aeroforge.geometry.operations`: cosine point distribution / repaneling.
- Stubbed public interfaces for the XFOIL solver, convergence strategies,
  optimization (pymoo), campaigns, and visualization layers.
- Continuous integration (lint, type-check, multi-OS test matrix) and
  documentation deployment workflows.

[Unreleased]: https://github.com/vcaries/airfoil_optimization/commits/main
