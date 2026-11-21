"""Combine several detectors into one decision."""

from __future__ import annotations

import logging

import numpy as np

from anomaly_detector.detectors.base import BaseDetector

logger = logging.getLogger(__name__)

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
        logger.info(
            "fitting %d detector(s) with strategy=%r: %s",
            len(self.detectors),
            self.strategy,
            ", ".join(d.name for d in self.detectors),
        )
        for det in self.detectors:
            det.fit(data)
        return self

    def votes(self, data) -> np.ndarray:
        """Matrix of shape (n_rows, n_detectors) with each detector's boolean flag."""
        return np.column_stack([det.predict(data) for det in self.detectors])

    def predict(self, data) -> np.ndarray:
        return self.combine(self.votes(data))

    def combine(self, votes: np.ndarray) -> np.ndarray:
        """Apply the voting strategy to a precomputed (n_rows, n_detectors) vote matrix."""
        votes = np.asarray(votes, dtype=bool)
        if self.strategy == "any":
            result = votes.any(axis=1)
        elif self.strategy == "all":
            result = votes.all(axis=1)
        else:
            result = votes.sum(axis=1) > (votes.shape[1] / 2)
        logger.info(
            "ensemble (%s) flagged %d/%d rows", self.strategy, int(result.sum()), len(result)
        )
        return result

    def fit_predict(self, data) -> np.ndarray:
        return self.fit(data).predict(data)
