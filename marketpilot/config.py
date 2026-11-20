"""Settings loader. Values come from config/settings.yaml; DATABASE_URL from env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS = ROOT / "config" / "settings.yaml"


@dataclass
class Settings:
    raw: dict[str, Any]
    root: Path = ROOT

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def account(self) -> dict:
        return self.raw["account"]

    def asset_class_of(self, symbol: str) -> str:
        for cls, spec in self.raw["universe"].items():
            if symbol in spec["symbols"]:
                return cls
        raise KeyError(f"{symbol} is not in the configured universe")

    def interval_of(self, symbol: str) -> str:
        return self.raw["universe"][self.asset_class_of(symbol)]["interval"]

    def all_symbols(self) -> list[str]:
        return [s for spec in self.raw["universe"].values() for s in spec["symbols"]]


def load_settings(path: str | Path | None = None, overrides: dict | None = None) -> Settings:
    path = Path(path or os.environ.get("MARKETPILOT_SETTINGS", DEFAULT_SETTINGS))
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if overrides:
        _deep_update(raw, overrides)
    return Settings(raw=raw)


def _deep_update(base: dict, upd: dict) -> None:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
