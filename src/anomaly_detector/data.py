"""Data loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUPPORTED = {".csv", ".tsv", ".json", ".parquet"}


def load_data(path: str | Path, timestamp_column: str | None = None) -> pd.DataFrame:
    """Read a CSV/TSV/JSON/Parquet file into a DataFrame.

    If ``timestamp_column`` is given it is parsed as datetimes and the frame is sorted by it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported file type {suffix!r}; use one of {sorted(SUPPORTED)}")

    if timestamp_column:
        if timestamp_column not in df.columns:
            raise KeyError(f"timestamp column {timestamp_column!r} not in data")
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df = df.sort_values(timestamp_column).reset_index(drop=True)
    return df


def select_columns(df: pd.DataFrame, columns: list[str] | None) -> pd.DataFrame:
    """Pick the numeric columns to analyse (all numeric columns when ``columns`` is None)."""
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"columns not found in data: {missing}")
        subset = df[columns]
        non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(subset[c])]
        if non_numeric:
            raise ValueError(f"columns are not numeric: {non_numeric}")
        return subset
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("no numeric columns found in data")
    return numeric


def generate_synthetic(
    n: int = 500,
    anomaly_fraction: float = 0.02,
    seed: int = 7,
) -> pd.DataFrame:
    """Create a sensor-like time series with injected spikes. Labels are in ``is_anomaly``."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    temperature = 20 + 2 * np.sin(2 * np.pi * t / 96) + rng.normal(0, 0.4, n)
    pressure = 1013 + rng.normal(0, 1.5, n)
    humidity = 45 + 5 * np.cos(2 * np.pi * t / 96) + rng.normal(0, 1.0, n)

    labels = np.zeros(n, dtype=bool)
    k = max(1, int(round(n * anomaly_fraction)))
    idx = rng.choice(np.arange(10, n), size=k, replace=False)
    labels[idx] = True
    sign = rng.choice([-1, 1], size=k)
    temperature[idx] += sign * rng.uniform(6, 10, k)
    pressure[idx] += sign * rng.uniform(10, 20, k)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "temperature": temperature.round(3),
            "pressure": pressure.round(3),
            "humidity": humidity.round(3),
            "is_anomaly": labels.astype(int),
        }
    )
