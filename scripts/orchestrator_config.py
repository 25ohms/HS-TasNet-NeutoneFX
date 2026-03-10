#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_orchestrator_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_nested(config: Dict[str, Any], *keys: str) -> Any:
    cursor: Any = config
    for key in keys:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
        if cursor is None:
            return None
    return cursor
