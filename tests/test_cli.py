import json
from pathlib import Path

import pandas as pd

from anomaly_detector.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_list_detectors(capsys):
    assert main(["list-detectors"]) == 0
    out = capsys.readouterr().out
    for name in ("zscore", "iqr", "mad", "rolling_zscore", "isolation_forest"):
        assert name in out


def test_generate(tmp_path, capsys):
    out = tmp_path / "gen.csv"
    assert main(["generate", "-o", str(out), "--rows", "100"]) == 0
    assert len(pd.read_csv(out)) == 100


def test_detect_with_config_and_output(tmp_path, csv_file, capsys):
    out = tmp_path / "scored.csv"
    code = main(
        [
            "detect",
            str(csv_file),
            "-c",
            str(ROOT / "configs" / "default.yaml"),
            "-o",
            str(out),
            "--label-column",
            "is_anomaly",
            "--json",
        ]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["rows"] == 400
    assert summary["evaluation"]["recall"] >= 0.8
    scored = pd.read_csv(out)
    assert "is_anomaly_pred" in scored.columns


def test_detect_cli_overrides(tmp_path, csv_file, capsys):
    out = tmp_path / "only.csv"
    code = main(
        [
            "detect",
            str(csv_file),
            "-d",
            "zscore",
            "-d",
            "mad",
            "--columns",
            "temperature",
            "pressure",
            "--strategy",
            "any",
            "--only-anomalies",
            "-o",
            str(out),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "Strategy      : any" in text
    flagged = pd.read_csv(out)
    assert flagged["is_anomaly_pred"].all()


def test_label_column_excluded_from_features(csv_file, capsys):
    assert main(["detect", str(csv_file), "--label-column", "is_anomaly", "--json"]) == 0
    assert "evaluation" in json.loads(capsys.readouterr().out)


def test_missing_file_returns_error_code(capsys):
    assert main(["detect", "nope.csv"]) == 2
    assert "error" in capsys.readouterr().err


def test_bad_column_returns_error_code(csv_file, capsys):
    assert main(["detect", str(csv_file), "--columns", "nope"]) == 2
