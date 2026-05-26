# Contributing to aeroforge

Thanks for your interest in the project. This guide describes the development
workflow and the quality bar the codebase holds itself to.

## Development setup

```bash
git clone https://github.com/vcaries/airfoil_optimization.git
cd airfoil_optimization
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make dev          # editable install + dev deps + pre-commit hooks
```

To run the XFOIL integration tests you also need the `xfoil` binary on your
`PATH` (see the README for installation notes).

## Branching model

We use a lightweight [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow):

- `main` is always releasable and protected.
- Work happens on short-lived branches named `feat/...`, `fix/...`,
  `docs/...`, `refactor/...`, or `test/...`.
- Open a pull request early; CI must be green before merge.

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(geometry): add CST parametric generator
fix(solver): retry on non-convergence with reduced step
docs(readme): document XFOIL installation on Windows
```

This keeps the history readable and enables automated changelog generation.

## Quality gates

Every change must pass, locally and in CI:

```bash
make lint     # ruff
make type     # mypy
make test     # pytest (unit)
```

- **Style**: PEP 8 enforced by `ruff` + `black`-compatible formatting,
  100-char lines.
- **Docstrings**: Google style, required on all public modules, classes, and
  functions (`ruff` rule set `D`).
- **Typing**: full type hints on public APIs; `mypy` runs in strict-ish mode.
- **Tests**: new behavior ships with tests. Aim to keep coverage from
  regressing. Pure-geometry / parsing logic should be deterministic and
  covered by fast unit tests; anything touching the XFOIL binary belongs
  behind the `integration` marker.

## Adding new capabilities

The architecture is built around extension points (abstract base classes and
registries). Prefer adding a new strategy/generator/objective subclass over
modifying existing ones (open/closed principle). See `ARCHITECTURE.md` for the
relevant base class of each subsystem.
