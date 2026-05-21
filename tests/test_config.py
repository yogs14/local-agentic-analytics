from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core import config


def test_load_config_reads_yaml_from_configs_dir(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "app.yaml").write_text(
        "app:\n  name: local-agentic-analytics\n", encoding="utf-8"
    )
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)

    loaded = config.load_config("app.yaml")

    assert loaded == {"app": {"name": "local-agentic-analytics"}}


def test_load_config_raises_file_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        config.load_config("missing.yaml")


def test_load_all_configs_reads_expected_files(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    expected_files = ("app.yaml", "model.yaml", "duckdb.yaml", "chromadb.yaml")
    for file_name in expected_files:
        section_name = Path(file_name).stem
        (config_dir / file_name).write_text(
            f"{section_name}:\n  enabled: true\n", encoding="utf-8"
        )
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)

    loaded = config.load_all_configs()

    assert set(loaded) == {"app", "model", "duckdb", "chromadb"}
    assert loaded["duckdb"] == {"duckdb": {"enabled": True}}
