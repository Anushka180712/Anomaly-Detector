# Anomaly Detector

[![CI](https://github.com/Anushka180712/Anomaly-Detector/actions/workflows/ci.yml/badge.svg)](https://github.com/Anushka180712/Anomaly-Detector/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A small, configurable toolkit for finding anomalies (outliers) in tabular and
time-series data. It ships five detectors, lets you combine them with voting,
and can be used from the command line or as a Python library.

## Features

- **Five detectors** behind one interface (`fit` / `score` / `predict`)
  | Name | Method | Good for |
  |------|--------|----------|
  | `zscore` | Distance from the mean in standard deviations | Roughly normal data |
  | `iqr` | Tukey fences (distance outside Q1–Q3 in IQR units) | Skewed data, no normality assumption |
  | `mad` | Modified z-score using the median absolute deviation | Data that already contains big outliers |
  | `rolling_zscore` | Z-score against a trailing window | Time series with trends or level shifts |
  | `isolation_forest` | scikit-learn Isolation Forest on scaled features | Multivariate / non-linear patterns |
- **Ensemble voting**: `any`, `all` or `majority`
- **YAML / JSON configuration** with validation and clear error messages
- **CSV, TSV, JSON and Parquet** input
- **Evaluation**: pass a ground-truth column to get precision, recall and F1
- **Synthetic data generator** for experiments
- Multi-column data, missing values and constant columns are handled safely
- Tests, linting and GitHub Actions CI across Python 3.10–3.13

## Project structure

```
Anomaly-Detector/
├── src/anomaly_detector/
│   ├── detectors/
│   │   ├── base.py           # BaseDetector interface
│   │   ├── statistical.py    # zscore, iqr, mad
│   │   ├── timeseries.py     # rolling_zscore
│   │   ├── ml.py             # isolation_forest
│   │   └── __init__.py       # registry + get_detector()
│   ├── ensemble.py           # voting across detectors
│   ├── config.py             # YAML/JSON config loading and validation
│   ├── data.py               # file loading, column selection, synthetic data
│   ├── pipeline.py           # run_pipeline() and evaluate()
│   └── cli.py                # command-line interface
├── configs/                  # default.yaml, strict.yaml
├── data/sample_sensor.csv    # 500-row labelled sample dataset
├── examples/quickstart.py    # library usage example
├── scripts/                  # helper scripts
├── tests/                    # pytest suite
├── .github/workflows/ci.yml  # lint + test matrix
├── Makefile
└── pyproject.toml
```

## Setup

Requirements: Python 3.10 or newer and `git`.

```bash
git clone https://github.com/Anushka180712/Anomaly-Detector.git
cd Anomaly-Detector

# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install the package in editable mode with dev tools
pip install -e ".[dev]"
```

Prefer plain requirements files? `pip install -r requirements-dev.txt` and then
`pip install -e .` does the same thing.

Check it worked:

```bash
anomaly-detector --version
anomaly-detector list-detectors
```

## Usage

### Command line

Run every detector in the default config on the bundled sample data and compare
the result against its labels:

```bash
anomaly-detector detect data/sample_sensor.csv \
  -c configs/default.yaml \
  --label-column is_anomaly \
  -o output/results.csv
```

Example output:

```
Rows analysed : 500
Anomalies     : 10 (2.00%)
Strategy      : majority
  - zscore             10 flagged
  - iqr                12 flagged
  - mad                10 flagged
  - rolling_zscore     12 flagged
  - isolation_forest   10 flagged
Precision 1.000 | Recall 1.000 | F1 1.000
Results written to output/results.csv
```

The output CSV keeps every original column and adds `<detector>_score`,
`<detector>_flag`, `votes` and `is_anomaly_pred`.

More examples:

```bash
# pick detectors and columns on the command line (no config file needed)
anomaly-detector detect data/sample_sensor.csv -d zscore -d mad --columns temperature pressure

# flag a row if any detector fires, and only save flagged rows
anomaly-detector detect data/sample_sensor.csv -c configs/default.yaml --strategy any \
  --only-anomalies -o output/anomalies.csv

# machine-readable summary
anomaly-detector detect data/sample_sensor.csv --json

# generate a new synthetic dataset
anomaly-detector generate -o data/my_data.csv --rows 2000 --anomaly-fraction 0.03

# also works without the console script
python -m anomaly_detector detect data/sample_sensor.csv
```

Run `anomaly-detector detect --help` for every option. The CLI exits with code
`2` and a readable message on bad input (missing file, unknown column, invalid
config).

### Configuration

```yaml
timestamp_column: timestamp      # optional: parsed as datetime and sorted
columns: [temperature, pressure] # optional: defaults to all numeric columns
strategy: majority               # any | all | majority
dropna: false
detectors:
  - name: zscore
    threshold: 3.0
  - name: rolling_zscore
    threshold: 3.5
    window: 48
  - name: isolation_forest
    contamination: 0.02
```

Detector parameters can be written inline (as above) or under a `params:` key.
Command-line flags override values from the config file.

| Detector | Parameters (defaults) |
|----------|-----------------------|
| `zscore` | `threshold=3.0` |
| `iqr` | `threshold=1.5` |
| `mad` | `threshold=3.5` |
| `rolling_zscore` | `threshold=3.0`, `window=30`, `min_periods=window/2` |
| `isolation_forest` | `threshold=0.0`, `contamination="auto"`, `n_estimators=200`, `random_state=42` |

### Python library

```python
from anomaly_detector import EnsembleDetector, MADDetector, ZScoreDetector, IsolationForestDetector
from anomaly_detector.data import load_data

df = load_data("data/sample_sensor.csv", timestamp_column="timestamp")
X = df[["temperature", "pressure", "humidity"]]

ensemble = EnsembleDetector(
    [ZScoreDetector(), MADDetector(), IsolationForestDetector(contamination=0.02)],
    strategy="majority",
)
df["anomaly"] = ensemble.fit_predict(X)
print(df[df["anomaly"]])
```

Or run the whole config-driven pipeline:

```python
from anomaly_detector import run_pipeline
from anomaly_detector.config import load_config

result = run_pipeline(df, load_config("configs/default.yaml"))
print(result.summary())
result.anomalies.to_csv("output/anomalies.csv", index=False)
```

See [`examples/quickstart.py`](examples/quickstart.py) for a runnable script.

### Adding a detector

1. Subclass `BaseDetector`, set a unique `name`, and implement `fit()` and
   `score()` (higher score means more anomalous).
2. Register the class in `DETECTORS` in `src/anomaly_detector/detectors/__init__.py`.
3. Add tests in `tests/test_detectors.py`.

It is then available in configs and via `--detector`.

## Testing

```bash
# run the full suite
pytest

# with coverage
pytest --cov=anomaly_detector --cov-report=term-missing

# a single file or test
pytest tests/test_detectors.py
pytest tests/test_cli.py::test_generate -v

# lint and format checks (the same ones CI runs)
ruff check .
ruff format --check .
```

With `make` available: `make install`, `make test`, `make lint`, `make run`.

What the tests cover:

| File | Covers |
|------|--------|
| `test_detectors.py` | Each detector finds injected outliers, low false-positive rate on clean data, NaN and constant-column handling, registry |
| `test_ensemble.py` | `any` / `all` / `majority` voting and validation |
| `test_config.py` | YAML/JSON parsing, invalid configs, shipped configs stay valid |
| `test_pipeline.py` | Data loading for each format, column selection, end-to-end accuracy, metrics |
| `test_cli.py` | Every subcommand, flag overrides, error exit codes |

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and on pull requests:
`ruff` lint and format checks, then the test suite with coverage on Python
3.10, 3.11, 3.12 and 3.13, followed by a CLI smoke test on the sample data.

## Regenerating the sample data

```bash
python scripts/generate_sample_data.py --rows 500 --seed 7
```

## License

[MIT](LICENSE) © Udisha Singh
