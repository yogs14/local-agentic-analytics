"""YAML-backed dataset profile loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from local_agentic_analytics.core.config import PROJECT_ROOT


DOMAINS_DIR = PROJECT_ROOT / "domains"


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    type: str
    unit: str = ""
    semantic_type: str = ""
    description: str = ""
    rules: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    domain: str
    table_name: str
    datetime_column: str
    sampling_frequency: str = ""
    description: str = ""
    columns: dict[str, ColumnProfile] = field(default_factory=dict)
    sql_rules: dict[str, str] = field(default_factory=dict)


def load_dataset_profile(domain: str = "energy") -> DatasetProfile:
    """Load a dataset profile from ``domains/{domain}/profile.yaml``."""
    if not domain or not domain.strip():
        raise ValueError("domain must not be empty")

    domain_name = domain.strip()
    domain_dir = DOMAINS_DIR / domain_name
    if not domain_dir.is_dir():
        raise FileNotFoundError(f"Domain not found: {domain_dir}")

    profile_path = domain_dir / "profile.yaml"
    if not profile_path.is_file():
        raise FileNotFoundError(f"Dataset profile file not found: {profile_path}")

    try:
        raw_profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid dataset profile YAML: {profile_path}") from exc

    if not isinstance(raw_profile, dict):
        raise ValueError(f"Dataset profile must contain a YAML mapping: {profile_path}")

    return _parse_dataset_profile(raw_profile, profile_path)


def profile_to_prompt_context(profile: DatasetProfile) -> str:
    """Convert a dataset profile into detailed prompt context."""
    lines = [
        f"Dataset name: {profile.name}",
        f"Domain: {profile.domain}",
        f"Canonical table name: {profile.table_name}",
        f"Datetime column: {profile.datetime_column}",
    ]
    if profile.sampling_frequency:
        lines.append(f"Sampling frequency: {profile.sampling_frequency}")
    if profile.description:
        lines.append(f"Description: {profile.description}")

    lines.append("")
    lines.append("Columns:")
    for column in profile.columns.values():
        details = [column.type]
        if column.unit:
            details.append(f"unit={column.unit}")
        if column.semantic_type:
            details.append(f"semantic_type={column.semantic_type}")
        if column.description:
            details.append(column.description)
        lines.append(f"- {column.name}: " + "; ".join(details))

        if column.rules:
            for rule_name, rule_value in column.rules.items():
                lines.append(f"  {rule_name}: {rule_value}")

    lines.append("")
    lines.append("SQL rules:")
    for rule_name, rule_value in profile.sql_rules.items():
        lines.append(f"- {rule_name}: {rule_value}")

    return "\n".join(lines).strip()


def profile_to_compact_sql_context(profile: DatasetProfile) -> str:
    """Convert a dataset profile into a compact SQL-only prompt context."""
    columns = ", ".join(profile.columns.keys())
    unit_parts = [
        f"{column.name}={column.unit}"
        for column in profile.columns.values()
        if column.unit
    ]
    units = ", ".join(unit_parts)
    date_filter = profile.sql_rules.get(
        "date_filter",
        f"CAST({profile.datetime_column} AS DATE) = DATE 'YYYY-MM-DD'",
    )
    missing_value_count = profile.sql_rules.get(
        "missing_value_count",
        "COUNT(*) FILTER (WHERE <column> IS NULL)",
    )

    lines = [
        f"table_name: {profile.table_name}",
        f"datetime_column: {profile.datetime_column}",
        f"available_columns: {columns}",
    ]
    if units:
        lines.append(f"important_units: {units}")

    lines.extend(
        [
            "sql_rules:",
            "- AVG(Global_active_power) AS avg_global_active_power_kw",
            "- SUM(Global_active_power) / 60.0 AS total_energy_kwh",
            f"- {missing_value_count}",
            f"- {date_filter}",
        ]
    )

    return "\n".join(lines).strip()


def _parse_dataset_profile(raw_profile: dict[str, Any], profile_path: Path) -> DatasetProfile:
    dataset = raw_profile.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"Dataset profile missing 'dataset' mapping: {profile_path}")

    table_name = _required_string(dataset, "table_name", profile_path)
    datetime_column = _required_string(dataset, "datetime_column", profile_path)

    raw_columns = raw_profile.get("columns", {})
    if not isinstance(raw_columns, dict):
        raise ValueError(f"Dataset profile 'columns' must be a mapping: {profile_path}")

    columns = {
        column_name: _parse_column_profile(column_name, column_data, profile_path)
        for column_name, column_data in raw_columns.items()
    }

    raw_sql_rules = raw_profile.get("sql_rules", {})
    if not isinstance(raw_sql_rules, dict):
        raise ValueError(f"Dataset profile 'sql_rules' must be a mapping: {profile_path}")

    return DatasetProfile(
        name=str(dataset.get("name", "")).strip(),
        domain=str(dataset.get("domain", "")).strip(),
        table_name=table_name,
        datetime_column=datetime_column,
        sampling_frequency=str(dataset.get("sampling_frequency", "")).strip(),
        description=str(dataset.get("description", "")).strip(),
        columns=columns,
        sql_rules={
            str(rule_name): str(rule_value)
            for rule_name, rule_value in raw_sql_rules.items()
        },
    )


def _parse_column_profile(
    column_name: str,
    column_data: Any,
    profile_path: Path,
) -> ColumnProfile:
    if not isinstance(column_data, dict):
        raise ValueError(
            f"Column profile for '{column_name}' must be a mapping: {profile_path}"
        )

    raw_rules = column_data.get("rules", {})
    if raw_rules is None:
        raw_rules = {}
    if not isinstance(raw_rules, dict):
        raise ValueError(
            f"Column rules for '{column_name}' must be a mapping: {profile_path}"
        )

    return ColumnProfile(
        name=str(column_name),
        type=str(column_data.get("type", "")).strip(),
        unit=str(column_data.get("unit", "")).strip(),
        semantic_type=str(column_data.get("semantic_type", "")).strip(),
        description=str(column_data.get("description", "")).strip(),
        rules={str(rule_name): str(rule_value) for rule_name, rule_value in raw_rules.items()},
    )


def _required_string(mapping: dict[str, Any], key: str, profile_path: Path) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Dataset profile '{key}' must not be empty: {profile_path}")
    return value.strip()
