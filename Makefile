.PHONY: help install install-dev test lint format clean docs serve-docs build upload

help:
	@echo "Available commands:"
	@echo "  install       Install package in production mode"
	@echo "  install-dev   Install package in development mode"
	@echo "  test          Run tests with coverage"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with black and isort"
	@echo "  clean         Clean build artifacts"
	@echo "  docs          Build documentation"
	@echo "  serve-docs    Serve documentation locally"
	@echo "  build         Build distribution packages"
	@echo "  upload        Upload to PyPI"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest --cov=unified_xai --cov-report=html --cov-report=term

lint:
	flake8 unified_xai tests
	pylint unified_xai
	mypy unified_xai
	black --check unified_xai tests
	isort --check-only unified_xai tests

format:
	black unified_xai tests
	isort unified_xai tests

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	cd docs && make clean && make html

serve-docs:
	cd docs/_build/html && python -m http.server

build: clean
	python -m build

upload: build
	twine check dist/*
	twine upload dist/*

# Development workflow
dev: install-dev lint test

# CI simulation
ci: lint test build