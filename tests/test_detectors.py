import numpy as np
import pandas as pd
import pytest

from anomaly_detector.detectors import (
    DETECTORS,
    IQRDetector,
    IsolationForestDetector,
    MADDetector,
    RollingZScoreDetector,
    ZScoreDetector,
    get_detector,
)

STAT_DETECTORS = [ZScoreDetector, IQRDetector, MADDetector]


@pytest.mark.parametrize("cls", STAT_DETECTORS)
def test_statistical_detectors_find_injected_outliers(cls, series_with_outliers):
    series, expected = series_with_outliers
    flags = cls().fit_predict(series)
    assert expected.issubset(set(np.flatnonzero(flags)))


@pytest.mark.parametrize("cls", STAT_DETECTORS)
def test_clean_data_has_few_flags(cls):
    data = np.random.default_rng(1).normal(0, 1, 1000)
    flags = cls().fit_predict(data)
    assert flags.mean() < 0.02


@pytest.mark.parametrize("cls", [*STAT_DETECTORS, IsolationForestDetector])
def test_score_before_fit_raises(cls):
    with pytest.raises(RuntimeError):
        cls().score(np.arange(10.0))


def test_zscore_score_values():
    det = ZScoreDetector().fit(np.array([0.0, 0.0, 10.0, 10.0]))  # mean 5, std 5
    np.testing.assert_allclose(det.score(np.array([5.0, 15.0, -10.0])), [0.0, 2.0, 3.0])


def test_iqr_matches_tukey_rule():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 100], dtype=float)
    flags = IQRDetector(threshold=1.5).fit_predict(data)
    assert flags.tolist() == [False] * 8 + [True]


def test_mad_is_robust_to_extreme_outlier():
    data = np.r_[np.full(50, 10.0) + np.linspace(-1, 1, 50), 1e6]
    det = MADDetector().fit(data)
    assert abs(det.median_[0] - 10.0) < 0.1
    assert det.predict(data)[-1]


def test_constant_column_does_not_divide_by_zero():
    data = np.ones(20)
    for cls in STAT_DETECTORS:
        scores = cls().fit(data).score(data)
        assert np.all(np.isfinite(scores))
        assert not cls().fit_predict(data).any()


def test_multicolumn_score_uses_worst_column():
    df = pd.DataFrame({"a": np.zeros(100), "b": np.random.default_rng(2).normal(0, 1, 100)})
    df.loc[10, "b"] = 50
    flags = ZScoreDetector().fit_predict(df)
    assert flags[10]


def test_nan_values_are_tolerated():
    data = np.r_[np.random.default_rng(3).normal(0, 1, 100), np.nan, 40.0]
    for cls in STAT_DETECTORS:
        flags = cls().fit_predict(data)
        assert not flags[100] and flags[101]


def test_rolling_zscore_ignores_level_shift_but_catches_spike():
    rng = np.random.default_rng(4)
    values = np.r_[rng.normal(0, 1, 200), rng.normal(5, 1, 200)]
    values[300] = 20
    det = RollingZScoreDetector(threshold=4, window=30)
    flags = det.fit_predict(values)
    assert flags[300]
    # A global z-score would flag lots of the shifted segment; rolling should not.
    assert flags.sum() <= 5


def test_rolling_window_validation():
    with pytest.raises(ValueError):
        RollingZScoreDetector(window=1)


def test_isolation_forest_flags_far_points():
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (300, 2))
    X[:5] = [[8, 8], [-8, 8], [8, -8], [-8, -8], [10, 0]]
    flags = IsolationForestDetector(contamination=0.02).fit_predict(X)
    assert flags[:5].all()


def test_isolation_forest_is_deterministic_with_seed():
    X = np.random.default_rng(6).normal(0, 1, (100, 3))
    a = IsolationForestDetector(random_state=1).fit(X).score(X)
    b = IsolationForestDetector(random_state=1).fit(X).score(X)
    np.testing.assert_allclose(a, b)


def test_invalid_threshold():
    with pytest.raises(ValueError):
        ZScoreDetector(threshold=0)


def test_non_numeric_input_rejected():
    with pytest.raises(ValueError):
        ZScoreDetector().fit(pd.DataFrame({"x": ["a", "b"]}))


def test_registry():
    assert set(DETECTORS) == {"zscore", "iqr", "mad", "rolling_zscore", "isolation_forest"}
    assert isinstance(get_detector("zscore", threshold=2), ZScoreDetector)
    with pytest.raises(ValueError, match="unknown detector"):
        get_detector("nope")
