import numpy as np
import pandas as pd
import pytest

from anomaly_detector.config import Config, DetectorSpec
from anomaly_detector.data import generate_synthetic, load_data, select_columns
from anomaly_detector.pipeline import evaluate, run_pipeline


def test_generate_synthetic_shape_and_labels():
    df = generate_synthetic(n=200, anomaly_fraction=0.05, seed=1)
    assert len(df) == 200
    assert df["is_anomaly"].sum() == 10
    assert {"timestamp", "temperature", "pressure", "humidity"} <= set(df.columns)


def test_generate_is_reproducible():
    pd.testing.assert_frame_equal(generate_synthetic(seed=3), generate_synthetic(seed=3))


def test_load_csv_sorts_by_timestamp(tmp_path):
    df = pd.DataFrame({"ts": ["2024-01-02", "2024-01-01"], "v": [2, 1]})
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    out = load_data(p, timestamp_column="ts")
    assert out["v"].tolist() == [1, 2]
    assert pd.api.types.is_datetime64_any_dtype(out["ts"])


def test_load_other_formats(tmp_path, synthetic_df):
    tsv = tmp_path / "d.tsv"
    synthetic_df.to_csv(tsv, sep="\t", index=False)
    assert len(load_data(tsv)) == len(synthetic_df)
    js = tmp_path / "d.json"
    synthetic_df.drop(columns="timestamp").to_json(js, orient="records")
    assert len(load_data(js)) == len(synthetic_df)


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "missing.csv")
    bad = tmp_path / "d.xlsx"
    bad.write_text("x")
    with pytest.raises(ValueError):
        load_data(bad)
    ok = tmp_path / "d.csv"
    ok.write_text("a\n1\n")
    with pytest.raises(KeyError):
        load_data(ok, timestamp_column="nope")


def test_select_columns():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1.0, 2.0]})
    assert list(select_columns(df, None).columns) == ["a", "c"]
    assert list(select_columns(df, ["c"]).columns) == ["c"]
    with pytest.raises(KeyError):
        select_columns(df, ["zzz"])
    with pytest.raises(ValueError):
        select_columns(df, ["b"])


def test_pipeline_end_to_end(synthetic_df):
    cfg = Config(
        columns=["temperature", "pressure", "humidity"],
        detectors=[DetectorSpec("zscore"), DetectorSpec("mad"), DetectorSpec("iqr")],
        strategy="majority",
    )
    result = run_pipeline(synthetic_df, cfg)
    for name in ("zscore", "mad", "iqr"):
        assert f"{name}_score" in result.data and f"{name}_flag" in result.data
    metrics = evaluate(synthetic_df["is_anomaly"], result.data["is_anomaly_pred"])
    assert metrics["recall"] >= 0.8
    assert metrics["precision"] >= 0.8
    summary = result.summary()
    assert summary["rows"] == len(synthetic_df)
    assert summary["anomalies"] == len(result.anomalies)


def test_pipeline_duplicate_detectors_get_unique_columns(synthetic_df):
    cfg = Config(
        columns=["temperature"],
        detectors=[DetectorSpec("zscore"), DetectorSpec("zscore", {"threshold": 2})],
    )
    result = run_pipeline(synthetic_df, cfg)
    assert result.detector_names == ["zscore", "zscore_2"]


def test_pipeline_dropna():
    df = pd.DataFrame({"v": [1.0, np.nan, 2.0, 1.5]})
    result = run_pipeline(df, Config(dropna=True))
    assert len(result.data) == 3


def test_evaluate():
    m = evaluate([1, 1, 0, 0], [1, 0, 1, 0])
    assert m == {
        "true_positives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert evaluate([0, 0], [0, 0])["f1"] == 0.0
