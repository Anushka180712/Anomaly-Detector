"""Combine several detectors into one decision."""

from __future__ import annotations

import numpy as np

from anomaly_detector.detectors.base import BaseDetector

STRATEGIES = ("any", "all", "majority")


class EnsembleDetector:
    """Vote across detectors.

    ``strategy``:
      * ``any``      - flag a row if at least one detector flags it (high recall)
      * ``all``      - flag only if every detector agrees (high precision)
      * ``majority`` - flag if more than half of the detectors agree
    """

    def __init__(self, detectors: list[BaseDetector], strategy: str = "majority") -> None:
        if not detectors:
            raise ValueError("ensemble needs at least one detector")
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        self.detectors = detectors
        self.strategy = strategy

    def fit(self, data) -> EnsembleDetector:
        for det in self.detectors:
            det.fit(data)
        return self

    def votes(self, data) -> np.ndarray:
        """Matrix of shape (n_rows, n_detectors) with each detector's boolean flag."""
        return np.column_stack([det.predict(data) for det in self.detectors])

    def predict(self, data) -> np.ndarray:
        votes = self.votes(data)
        if self.strategy == "any":
            return votes.any(axis=1)
        if self.strategy == "all":
            return votes.all(axis=1)
        return votes.sum(axis=1) > (votes.shape[1] / 2)

    def fit_predict(self, data) -> np.ndarray:
        return self.fit(data).predict(data)
