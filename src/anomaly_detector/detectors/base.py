"""Common interface shared by every detector."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """Abstract detector.

    Subclasses implement :meth:`fit` and :meth:`score`. Scores are non-negative and
    larger means "more anomalous". A point is flagged when its score exceeds
    ``self.threshold``.
    """

    name: str = "base"

    def __init__(self, threshold: float = 3.0) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = float(threshold)
        self._fitted = False

    @staticmethod
    def _to_frame(data: pd.DataFrame | pd.Series | np.ndarray) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            frame = data
        elif isinstance(data, pd.Series):
            frame = data.to_frame()
        else:
            arr = np.asarray(data, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            frame = pd.DataFrame(arr)
        numeric = frame.select_dtypes(include="number")
        if numeric.shape[1] == 0:
            raise ValueError("input contains no numeric columns")
        return numeric.astype(float)

    @abstractmethod
    def fit(self, data) -> BaseDetector:
        """Learn statistics from (mostly normal) data."""

    @abstractmethod
    def score(self, data) -> np.ndarray:
        """Return one anomaly score per row."""

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before scoring")

    def predict(self, data) -> np.ndarray:
        """Return a boolean mask, True where a row is anomalous."""
        scores = self.score(data)
        flags = scores > self.threshold
        logger.debug(
            "%s flagged %d/%d rows (threshold=%s, max score=%.3f)",
            self.name,
            int(flags.sum()),
            len(flags),
            self.threshold,
            float(scores.max()) if len(scores) else 0.0,
        )
        return flags

    def fit_predict(self, data) -> np.ndarray:
        return self.fit(data).predict(data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(threshold={self.threshold})"
