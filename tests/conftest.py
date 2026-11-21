import numpy as np
import pandas as pd
import pytest

from anomaly_detector.data import generate_synthetic


@pytest.fixture
def series_with_outliers():
    rng = np.random.default_rng(0)
    values = rng.normal(10, 1, 200)
    values[[50, 120, 180]] = [25, -5, 30]
    return pd.Series(values, name="value"), {50, 120, 180}


@pytest.fixture
def synthetic_df():
    return generate_synthetic(n=400, anomaly_fraction=0.02, seed=11)


@pytest.fixture
def csv_file(tmp_path, synthetic_df):
    path = tmp_path / "data.csv"
    synthetic_df.to_csv(path, index=False)
    return path
