# Developer convenience targets. Run `make help` to list them.
.DEFAULT_GOAL := help
.PHONY: help install dev lint format type test cov docs clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the package (runtime only)
	pip install -e .

dev: ## Install the package with all dev dependencies + pre-commit hooks
	pip install -e ".[dev]"
	pre-commit install

lint: ## Run ruff linter
	ruff check src tests

format: ## Auto-format the codebase
	ruff format src tests
	ruff check --fix src tests

type: ## Run mypy static type checker
	mypy src

test: ## Run the unit test suite
	pytest -m "not integration"

cov: ## Run tests with a coverage report
	pytest -m "not integration" --cov --cov-report=term-missing

docs: ## Serve the documentation locally
	mkdocs serve

clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
