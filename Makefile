.PHONY: help install dev-install test lint format check clean setup-dev

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the package
	pip install -e .

dev-install: ## Install package with dev dependencies
	pip install -e ".[dev]"

setup-dev: dev-install ## Set up development environment with pre-commit hooks
	pre-commit install

test: ## Run tests
	pytest tests/ -v

lint: ## Run comprehensive linting and formatting
	python3 -m autopep8 --in-place --aggressive --aggressive --max-line-length=88 --recursive .
	python3 -m ruff check . --fix --unsafe-fixes
	python3 -m ruff format .

format: ## Run ruff formatting
	python3 -m ruff format .

autopep8: ## Fix line length issues with autopep8
	python3 -m autopep8 --in-place --aggressive --aggressive --max-line-length=88 --recursive .

format-all: autopep8 format ## Run autopep8 + ruff formatting (comprehensive)

check: ## Run ruff checks without fixes (CI-style)
	python3 -m ruff check .
	python3 -m ruff format --check .

check-all: lint test ## Run all checks (lint + test)

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete 