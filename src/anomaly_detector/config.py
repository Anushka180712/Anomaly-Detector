"""Load and validate YAML / JSON configuration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from anomaly_detector.detectors import DETECTORS
from anomaly_detector.ensemble import STRATEGIES

logger = logging.getLogger(__name__)


@dataclass
class DetectorSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    columns: list[str] | None = None
    timestamp_column: str | None = None
    detectors: list[DetectorSpec] = field(
        default_factory=lambda: [DetectorSpec("zscore"), DetectorSpec("iqr")]
    )
    strategy: str = "majority"
    dropna: bool = False

    def validate(self) -> Config:
        if not self.detectors:
            raise ValueError("config must list at least one detector")
        for spec in self.detectors:
            if spec.name not in DETECTORS:
                raise ValueError(
                    f"unknown detector {spec.name!r} in config; "
                    f"choose from: {', '.join(sorted(DETECTORS))}"
                )
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {self.strategy!r}")
        return self


def _parse_detectors(raw: Any) -> list[DetectorSpec]:
    specs = []
    for item in raw or []:
        if isinstance(item, str):
            specs.append(DetectorSpec(item))
        elif isinstance(item, dict) and "name" in item:
            params = {k: v for k, v in item.items() if k != "name"}
            params.update(item.get("params", {}) or {})
            params.pop("params", None)
            specs.append(DetectorSpec(item["name"], params))
        else:
            raise ValueError(f"invalid detector entry: {item!r}")
    return specs


def config_from_dict(raw: dict[str, Any]) -> Config:
    raw = dict(raw or {})
    cfg = Config()
    if "columns" in raw:
        cfg.columns = raw["columns"]
    cfg.timestamp_column = raw.get("timestamp_column")
    if "detectors" in raw:
        cfg.detectors = _parse_detectors(raw["detectors"])
    cfg.strategy = raw.get("strategy", cfg.strategy)
    cfg.dropna = bool(raw.get("dropna", cfg.dropna))
    return cfg.validate()


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    cfg = config_from_dict(raw or {})
    logger.info(
        "loaded config %s: detectors=%s strategy=%s",
        path,
        [d.name for d in cfg.detectors],
        cfg.strategy,
    )
    return cfg
