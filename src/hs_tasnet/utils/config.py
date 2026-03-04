from __future__ import annotations

import copy
import pathlib
from typing import Any, Dict, Iterable

import yaml


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def load_config(path: str | pathlib.Path) -> Dict[str, Any]:
    path = pathlib.Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base = cfg.pop("base", None)
    if base:
        base_path = (path.parent / base).resolve()
        base_cfg = load_config(base_path)
        cfg = _deep_update(base_cfg, cfg)
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value.")
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return cfg


def save_config(cfg: Dict[str, Any], path: str | pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
