"""Regenerate data/sample_sensor.csv.

Usage: python scripts/generate_sample_data.py [--rows 500] [--seed 7]
"""

import argparse
from pathlib import Path

from anomaly_detector.data import generate_synthetic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="data/sample_sensor.csv")
    args = parser.parse_args()

    df = generate_synthetic(n=args.rows, seed=args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} rows to {args.output}")
