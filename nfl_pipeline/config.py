"""Configuration loading and canonical project paths."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "pipeline.yaml"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_DIR = ROOT / "predictions"
BEST_PARAMS_DIR = ROOT / "config" / "best_params"
APPROVAL_FILE = FEATURES_DIR / "APPROVED"

for _d in (RAW_DIR, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR, PREDICTIONS_DIR, BEST_PARAMS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Thin attribute-style wrapper over the YAML config dict."""

    raw: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - trivial
        raw = object.__getattribute__(self, "raw")
        if item in raw:
            val = raw[item]
            return Config(val) if isinstance(val, dict) else val
        raise AttributeError(item)

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw)

    # convenience -----------------------------------------------------------
    @property
    def seasons(self) -> list[int]:
        return list(range(int(self.get("data.first_season")), int(self.get("data.current_season")) + 1))

    @property
    def current_season(self) -> int:
        return int(self.get("data.current_season"))


def load_config(path: Path | str | None = None) -> Config:
    p = Path(path) if path else CONFIG_PATH
    with open(p, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh) or {})
