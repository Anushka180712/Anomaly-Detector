"""End-to-end detection pipeline used by the CLI and importable from Python."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from anomaly_detector.config import Config
from anomaly_detector.data import select_columns
from anomaly_detector.detectors import get_detector
from anomaly_detector.ensemble import EnsembleDetector

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    data: pd.DataFrame  # original rows plus score / flag columns
    detector_names: list[str]

    @property
    def anomalies(self) -> pd.DataFrame:
        return self.data[self.data["is_anomaly_pred"]]

    @property
    def n_anomalies(self) -> int:
        return int(self.data["is_anomaly_pred"].sum())

    def summary(self) -> dict:
        per_detector = {name: int(self.data[f"{name}_flag"].sum()) for name in self.detector_names}
        return {
            "rows": len(self.data),
            "anomalies": self.n_anomalies,
            "anomaly_rate": round(self.n_anomalies / max(len(self.data), 1), 4),
            "per_detector": per_detector,
        }


def run_pipeline(df: pd.DataFrame, config: Config) -> DetectionResult:
    """Fit every configured detector on ``df`` and combine their votes."""
    start = time.perf_counter()
    df = df.copy()
    if config.dropna:
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info("dropna removed %d rows with missing values", before - len(df))

    features = select_columns(df, config.columns)
    detectors = [get_detector(spec.name, **spec.params) for spec in config.detectors]
    ensemble = EnsembleDetector(detectors, strategy=config.strategy).fit(features)

    names = []
    flags = []
    for det in detectors:
        name = det.name
        # Disambiguate when the same detector is listed twice.
        suffix = 2
        while name in names:
            name = f"{det.name}_{suffix}"
            suffix += 1
        names.append(name)
        # Score once per detector and reuse it for both the score and the flag column.
        scores = det.score(features)
        flag = scores > det.threshold
        logger.debug("%s flagged %d/%d rows", name, int(flag.sum()), len(flag))
        df[f"{name}_score"] = np.round(scores, 6)
        df[f"{name}_flag"] = flag
        flags.append(flag)

    votes = np.column_stack(flags)
    df["votes"] = votes.sum(axis=1)
    df["is_anomaly_pred"] = ensemble.combine(votes)
    result = DetectionResult(data=df, detector_names=names)
    logger.info(
        "pipeline finished in %.2fs: %d anomalies in %d rows",
        time.perf_counter() - start,
        result.n_anomalies,
        len(df),
    )
    return result


def evaluate(y_true, y_pred) -> dict:
    """Precision, recall and F1 for boolean labels."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
