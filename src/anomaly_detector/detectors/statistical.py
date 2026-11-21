"""Classic statistical detectors: z-score, IQR and median absolute deviation."""

from __future__ import annotations

import logging

import numpy as np

from anomaly_detector.detectors.base import BaseDetector

logger = logging.getLogger(__name__)

_EPS = 1e-12


def _row_max(values: np.ndarray) -> np.ndarray:
    """Per-row maximum that treats missing values as non-anomalous (score 0)."""
    return np.nan_to_num(values, nan=0.0).max(axis=1)


class ZScoreDetector(BaseDetector):
    """Flags rows whose value lies more than ``threshold`` standard deviations from the mean.

    With several columns the row score is the maximum absolute z-score across columns.
    """

    name = "zscore"

    def __init__(self, threshold: float = 3.0) -> None:
        super().__init__(threshold)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, data) -> ZScoreDetector:
        frame = self._to_frame(data)
        self.mean_ = np.nanmean(frame.to_numpy(), axis=0)
        self.std_ = np.nanstd(frame.to_numpy(), axis=0)
        if np.any(self.std_ == 0):
            logger.warning("zscore: some columns have zero variance and will never be flagged")
        logger.debug("zscore fitted on %d rows: mean=%s std=%s", len(frame), self.mean_, self.std_)
        self._fitted = True
        return self

    def score(self, data) -> np.ndarray:
        self._check_fitted()
        values = self._to_frame(data).to_numpy()
        z = np.abs(values - self.mean_) / np.maximum(self.std_, _EPS)
        return _row_max(z)


class IQRDetector(BaseDetector):
    """Tukey-fence detector.

    The score is the distance outside the [Q1, Q3] box measured in IQR units, so a
    ``threshold`` of 1.5 reproduces the classic Tukey rule.
    """

    name = "iqr"

    def __init__(self, threshold: float = 1.5) -> None:
        super().__init__(threshold)
        self.q1_: np.ndarray | None = None
        self.q3_: np.ndarray | None = None

    def fit(self, data) -> IQRDetector:
        values = self._to_frame(data).to_numpy()
        self.q1_ = np.nanpercentile(values, 25, axis=0)
        self.q3_ = np.nanpercentile(values, 75, axis=0)
        logger.debug("iqr fitted on %d rows: q1=%s q3=%s", len(values), self.q1_, self.q3_)
        self._fitted = True
        return self

    def score(self, data) -> np.ndarray:
        self._check_fitted()
        values = self._to_frame(data).to_numpy()
        iqr = np.maximum(self.q3_ - self.q1_, _EPS)
        below = (self.q1_ - values) / iqr
        above = (values - self.q3_) / iqr
        dist = np.maximum(np.maximum(below, above), 0.0)
        return _row_max(dist)


class MADDetector(BaseDetector):
    """Robust z-score based on the median absolute deviation (Iglewicz & Hoaglin).

    The modified z-score is ``0.6745 * |x - median| / MAD``; 3.5 is the usual cut-off.
    """

    name = "mad"

    def __init__(self, threshold: float = 3.5) -> None:
        super().__init__(threshold)
        self.median_: np.ndarray | None = None
        self.mad_: np.ndarray | None = None

    def fit(self, data) -> MADDetector:
        values = self._to_frame(data).to_numpy()
        self.median_ = np.nanmedian(values, axis=0)
        self.mad_ = np.nanmedian(np.abs(values - self.median_), axis=0)
        if np.any(self.mad_ == 0):
            logger.warning("mad: MAD is zero for some columns; most values are identical")
        logger.debug(
            "mad fitted on %d rows: median=%s mad=%s", len(values), self.median_, self.mad_
        )
        self._fitted = True
        return self

    def score(self, data) -> np.ndarray:
        self._check_fitted()
        values = self._to_frame(data).to_numpy()
        z = 0.6745 * np.abs(values - self.median_) / np.maximum(self.mad_, _EPS)
        return _row_max(z)
