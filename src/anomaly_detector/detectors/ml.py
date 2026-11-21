"""Machine-learning based detectors."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from anomaly_detector.detectors.base import BaseDetector


class IsolationForestDetector(BaseDetector):
    """Wraps scikit-learn's Isolation Forest.

    Scores are ``-decision_function`` so that positive values are anomalous and the
    default ``threshold`` of 0 matches scikit-learn's own decision boundary. Features
    are standardised first so columns on different scales contribute equally.
    Missing values are filled with the training median.
    """

    name = "isolation_forest"

    def __init__(
        self,
        threshold: float = 0.0,
        contamination: float | str = "auto",
        n_estimators: int = 200,
        random_state: int | None = 42,
    ) -> None:
        # Bypass the positive-threshold check: 0 is a natural boundary here.
        self.threshold = float(threshold)
        self._fitted = False
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._model: IsolationForest | None = None
        self._fill: np.ndarray | None = None

    def _prepare(self, data) -> np.ndarray:
        values = self._to_frame(data).to_numpy()
        mask = np.isnan(values)
        if mask.any():
            values = np.where(mask, self._fill, values)
        return values

    def fit(self, data) -> IsolationForestDetector:
        values = self._to_frame(data).to_numpy()
        self._fill = np.nanmedian(values, axis=0)
        values = self._prepare(data)
        scaled = self._scaler.fit_transform(values)
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        ).fit(scaled)
        self._fitted = True
        return self

    def score(self, data) -> np.ndarray:
        self._check_fitted()
        scaled = self._scaler.transform(self._prepare(data))
        return -self._model.decision_function(scaled)
