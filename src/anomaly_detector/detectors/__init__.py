"""Detector registry."""

from __future__ import annotations

import logging

from anomaly_detector.detectors.base import BaseDetector
from anomaly_detector.detectors.ml import IsolationForestDetector
from anomaly_detector.detectors.statistical import IQRDetector, MADDetector, ZScoreDetector
from anomaly_detector.detectors.timeseries import RollingZScoreDetector

logger = logging.getLogger(__name__)

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
    try:
        detector = cls(**params)
    except TypeError as exc:
        # Unknown keyword in a config file: surface it as a readable config error.
        raise ValueError(f"invalid parameters for detector {name!r}: {exc}") from None
    logger.debug("created detector %r", detector)
    return detector


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
