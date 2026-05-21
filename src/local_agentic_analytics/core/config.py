"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"
CONFIG_FILES = ("app.yaml", "model.yaml", "duckdb.yaml", "chromadb.yaml")


def _resolve_config_path(path: str) -> Path:
    config_path = Path(path)

    if config_path.is_absolute():
        return config_path

    if config_path.parts and config_path.parts[0] == "configs":
        return PROJECT_ROOT / config_path

    return CONFIG_DIR / config_path


def load_config(path: str) -> dict[str, Any]:
    """Load one YAML config file.

    Relative paths are resolved from the project ``configs/`` directory.
    """
    config_path = _resolve_config_path(path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config file: {config_path}") from exc

    if content is None:
        return {}

    if not isinstance(content, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return content


def load_all_configs() -> dict[str, dict[str, Any]]:
    """Load all project configuration files keyed by config name."""
    return {Path(file_name).stem: load_config(file_name) for file_name in CONFIG_FILES}
