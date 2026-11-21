"""Anomaly Detector: a small, configurable toolkit for finding outliers in data."""

from anomaly_detector.detectors import (
    DETECTORS,
    BaseDetector,
    IQRDetector,
    IsolationForestDetector,
    MADDetector,
    RollingZScoreDetector,
    ZScoreDetector,
    get_detector,
)
from anomaly_detector.ensemble import EnsembleDetector
from anomaly_detector.pipeline import DetectionResult, run_pipeline

__version__ = "0.1.0"

__all__ = [
    "BaseDetector",
    "ZScoreDetector",
    "IQRDetector",
    "MADDetector",
    "RollingZScoreDetector",
    "IsolationForestDetector",
    "EnsembleDetector",
    "DETECTORS",
    "get_detector",
    "run_pipeline",
    "DetectionResult",
    "__version__",
]
