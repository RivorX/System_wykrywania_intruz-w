from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    base = base_dir or PROJECT_ROOT
    return (base / path).resolve()


def ensure_dir(path_value: str | Path, base_dir: Path | None = None) -> Path:
    directory = resolve_path(path_value, base_dir=base_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_yaml(path_value: str | Path, base_dir: Path | None = None) -> dict[str, Any]:
    path = resolve_path(path_value, base_dir=base_dir)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def save_yaml(path_value: str | Path, data: dict[str, Any], base_dir: Path | None = None) -> Path:
    path = resolve_path(path_value, base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
    return path


def save_json(path_value: str | Path, data: dict[str, Any], base_dir: Path | None = None) -> Path:
    path = resolve_path(path_value, base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    return path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
