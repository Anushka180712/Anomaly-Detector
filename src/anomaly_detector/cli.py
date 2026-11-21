"""Command-line interface.

Examples::

    anomaly-detector detect data/sample_sensor.csv -c configs/default.yaml -o out.csv
    anomaly-detector detect data.csv --detector zscore --detector iqr --strategy any
    anomaly-detector generate -o data/synthetic.csv --rows 1000
    anomaly-detector list-detectors
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from anomaly_detector import __version__
from anomaly_detector.config import Config, DetectorSpec, load_config
from anomaly_detector.data import generate_synthetic, load_data
from anomaly_detector.detectors import DETECTORS
from anomaly_detector.ensemble import STRATEGIES
from anomaly_detector.logging_config import DEFAULT_LEVEL, ENV_VAR, LEVELS, setup_logging
from anomaly_detector.pipeline import evaluate, run_pipeline

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anomaly-detector", description="Detect anomalies in tabular / time-series data."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", action="store_true", help="shortcut for --log-level DEBUG"
    )
    verbosity.add_argument("-q", "--quiet", action="store_true", help="only log errors")
    verbosity.add_argument(
        "--log-level",
        type=str.upper,
        choices=LEVELS,
        help=f"log level (default: ${ENV_VAR} or {DEFAULT_LEVEL})",
    )
    parser.add_argument("--log-file", help="also write logs to this file")
    sub = parser.add_subparsers(dest="command", required=True)

    det = sub.add_parser("detect", help="run detectors on a data file")
    det.add_argument("input", help="CSV / TSV / JSON / Parquet file")
    det.add_argument("-c", "--config", help="YAML or JSON config file")
    det.add_argument(
        "-d",
        "--detector",
        action="append",
        choices=sorted(DETECTORS),
        help="detector to use (repeatable); overrides the config",
    )
    det.add_argument("--columns", nargs="+", help="numeric columns to analyse")
    det.add_argument("--timestamp-column", help="column to parse as datetime and sort by")
    det.add_argument("--strategy", choices=STRATEGIES, help="ensemble voting strategy")
    det.add_argument("-o", "--output", help="write scored rows to this CSV")
    det.add_argument("--only-anomalies", action="store_true", help="output only flagged rows")
    det.add_argument("--label-column", help="ground-truth column (0/1) to report precision/recall")
    det.add_argument("--json", action="store_true", help="print the summary as JSON")

    gen = sub.add_parser("generate", help="create a synthetic labelled dataset")
    gen.add_argument("-o", "--output", required=True, help="output CSV path")
    gen.add_argument("--rows", type=int, default=500)
    gen.add_argument("--anomaly-fraction", type=float, default=0.02)
    gen.add_argument("--seed", type=int, default=7)

    sub.add_parser("list-detectors", help="show available detectors")
    return parser


def _resolve_config(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config) if args.config else Config()
    if args.detector:
        cfg.detectors = [DetectorSpec(name) for name in args.detector]
    if args.columns:
        cfg.columns = args.columns
    if args.timestamp_column:
        cfg.timestamp_column = args.timestamp_column
    if args.strategy:
        cfg.strategy = args.strategy
    return cfg.validate()


def _cmd_detect(args: argparse.Namespace) -> int:
    cfg = _resolve_config(args)
    df = load_data(args.input, timestamp_column=cfg.timestamp_column)

    if args.label_column and cfg.columns is None:
        # Never feed the ground truth to the detectors.
        cfg.columns = [
            c for c in df.select_dtypes(include="number").columns if c != args.label_column
        ]
        logger.info("excluding label column %r from features", args.label_column)

    result = run_pipeline(df, cfg)
    summary = result.summary()
    if args.label_column:
        if args.label_column not in result.data.columns:
            raise KeyError(f"label column {args.label_column!r} not in data")
        summary["evaluation"] = evaluate(
            result.data[args.label_column], result.data["is_anomaly_pred"]
        )

    if args.output:
        out = result.anomalies if args.only_anomalies else result.data
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        logger.info("wrote %d rows to %s", len(out), args.output)
        summary["output"] = str(args.output)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Rows analysed : {summary['rows']}")
        print(f"Anomalies     : {summary['anomalies']} ({summary['anomaly_rate']:.2%})")
        print(f"Strategy      : {cfg.strategy}")
        for name, count in summary["per_detector"].items():
            print(f"  - {name:<18} {count} flagged")
        if "evaluation" in summary:
            ev = summary["evaluation"]
            print(
                f"Precision {ev['precision']:.3f} | Recall {ev['recall']:.3f} | F1 {ev['f1']:.3f}"
            )
        if args.output:
            print(f"Results written to {args.output}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    df = generate_synthetic(args.rows, args.anomaly_fraction, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows ({int(df['is_anomaly'].sum())} anomalies) to {args.output}")
    return 0


def _cmd_list(_: argparse.Namespace) -> int:
    for name, cls in sorted(DETECTORS.items()):
        doc = (cls.__doc__ or "").strip().splitlines()[0]
        print(f"{name:<18} {doc}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    level = "DEBUG" if args.verbose else "ERROR" if args.quiet else args.log_level
    try:
        setup_logging(level, log_file=args.log_file)
    except ValueError as exc:  # bad ANOMALY_DETECTOR_LOG_LEVEL value
        print(f"error: {exc}", file=sys.stderr)
        return 2
    logger.debug("arguments: %s", vars(args))
    handlers = {"detect": _cmd_detect, "generate": _cmd_generate, "list-detectors": _cmd_list}
    try:
        return handlers[args.command](args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        logger.debug("command failed", exc_info=True)
        print(f"error: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
