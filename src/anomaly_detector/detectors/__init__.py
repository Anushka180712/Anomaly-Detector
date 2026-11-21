"""Detector registry."""

from __future__ import annotations

from anomaly_detector.detectors.base import BaseDetector
from anomaly_detector.detectors.ml import IsolationForestDetector
from anomaly_detector.detectors.statistical import IQRDetector, MADDetector, ZScoreDetector
from anomaly_detector.detectors.timeseries import RollingZScoreDetector

DETECTORS: dict[str, type[BaseDetector]] = {
    cls.name: cls
    for cls in (
        ZScoreDetector,
        IQRDetector,
        MADDetector,
        RollingZScoreDetector,
        IsolationForestDetector,
    )
}


def get_detector(name: str, **params) -> BaseDetector:
    """Instantiate a detector by its registry name."""
    try:
        cls = DETECTORS[name]
    except KeyError:
        known = ", ".join(sorted(DETECTORS))
        raise ValueError(f"unknown detector {name!r}; choose from: {known}") from None
    return cls(**params)


__all__ = [
    "BaseDetector",
    "ZScoreDetector",
    "IQRDetector",
    "MADDetector",
    "RollingZScoreDetector",
    "IsolationForestDetector",
    "DETECTORS",
    "get_detector",
]
