import json
from pathlib import Path

import pytest

from anomaly_detector.config import Config, config_from_dict, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_defaults():
    cfg = Config().validate()
    assert [d.name for d in cfg.detectors] == ["zscore", "iqr"]
    assert cfg.strategy == "majority"


def test_parse_mixed_detector_entries():
    cfg = config_from_dict(
        {
            "detectors": [
                "mad",
                {"name": "zscore", "threshold": 2.5},
                {"name": "rolling_zscore", "params": {"window": 10}},
            ],
            "strategy": "any",
        }
    )
    assert [d.name for d in cfg.detectors] == ["mad", "zscore", "rolling_zscore"]
    assert cfg.detectors[1].params == {"threshold": 2.5}
    assert cfg.detectors[2].params == {"window": 10}
    assert cfg.strategy == "any"


@pytest.mark.parametrize(
    "raw",
    [{"detectors": ["bogus"]}, {"strategy": "sometimes"}, {"detectors": []}, {"detectors": [3]}],
)
def test_invalid_configs(raw):
    with pytest.raises(ValueError):
        config_from_dict(raw)


def test_load_yaml_and_json(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("detectors:\n  - name: iqr\n    threshold: 2\nstrategy: all\n")
    assert load_config(y).detectors[0].params == {"threshold": 2}

    j = tmp_path / "c.json"
    j.write_text(json.dumps({"detectors": ["zscore"], "columns": ["a"]}))
    cfg = load_config(j)
    assert cfg.columns == ["a"]


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


@pytest.mark.parametrize("name", ["default.yaml", "strict.yaml"])
def test_shipped_configs_are_valid(name):
    load_config(ROOT / "configs" / name)
