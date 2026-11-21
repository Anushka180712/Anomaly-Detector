"""Use the library directly from Python.

Run from the repo root after `pip install -e .`:
    python examples/quickstart.py
"""

from anomaly_detector import EnsembleDetector, IsolationForestDetector, MADDetector, ZScoreDetector
from anomaly_detector.data import generate_synthetic
from anomaly_detector.pipeline import evaluate

df = generate_synthetic(n=1000, anomaly_fraction=0.02, seed=1)
features = df[["temperature", "pressure", "humidity"]]

# 1. A single detector
zscore = ZScoreDetector(threshold=3.0).fit(features)
print("z-score flagged:", int(zscore.predict(features).sum()))

# 2. An ensemble with majority voting
ensemble = EnsembleDetector(
    [ZScoreDetector(), MADDetector(), IsolationForestDetector(contamination=0.02)],
    strategy="majority",
)
pred = ensemble.fit_predict(features)
print("ensemble flagged:", int(pred.sum()))
print("metrics:", evaluate(df["is_anomaly"], pred))

# 3. Inspect the flagged rows
print(df[pred].head())
