.PHONY: install test lint format run data clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=anomaly_detector --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

run:
	anomaly-detector detect data/sample_sensor.csv -c configs/default.yaml --label-column is_anomaly -o output/results.csv

data:
	python scripts/generate_sample_data.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov output
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
