"""Time-series aware detectors."""

from __future__ import annotations

import numpy as np

from anomaly_detector.detectors.base import BaseDetector

_EPS = 1e-12


class RollingZScoreDetector(BaseDetector):
    """Z-score against a trailing rolling window.

    Each point is compared with the mean and standard deviation of the ``window``
    points *before* it, so level shifts and slow trends do not trigger alerts the way
    a global z-score would. The first ``min_periods`` points get a score of 0.
    """

    name = "rolling_zscore"

    def __init__(self, threshold: float = 3.0, window: int = 30, min_periods: int | None = None):
        super().__init__(threshold)
        if window < 2:
            raise ValueError("window must be at least 2")
        self.window = int(window)
        self.min_periods = int(min_periods) if min_periods else max(2, self.window // 2)

    def fit(self, data) -> RollingZScoreDetector:
        # Stateless: statistics are computed on the data being scored.
        self._to_frame(data)
        self._fitted = True
        return self

    def score(self, data) -> np.ndarray:
        self._check_fitted()
        frame = self._to_frame(data)
        rolling = frame.rolling(self.window, min_periods=self.min_periods)
        mean = rolling.mean().shift(1)
        std = rolling.std(ddof=0).shift(1)
        z = (frame - mean).abs() / std.clip(lower=_EPS)
        scores = z.max(axis=1, skipna=True).to_numpy(dtype=float)
        return np.nan_to_num(scores, nan=0.0)
