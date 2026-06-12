"""Dataset profile primitives for domain-aware analytics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    data_type: str
    unit: str = ""
    description: str = ""


@dataclass(frozen=True)
class SemanticRule:
    rule_id: str
    description: str
    sql_hint: str = ""


@dataclass(frozen=True)
class DatasetProfile:
    dataset_id: str
    display_name: str
    table_name: str
    datetime_column: str | None = None
    columns: tuple[ColumnProfile, ...] = field(default_factory=tuple)
    semantic_rules: tuple[SemanticRule, ...] = field(default_factory=tuple)

    def format_columns_for_prompt(self) -> str:
        if not self.columns:
            return "- No domain column metadata provided."

        lines = []
        for column in self.columns:
            details = [column.data_type]
            if column.unit:
                details.append(f"unit={column.unit}")
            if column.description:
                details.append(column.description)
            lines.append(f"- {column.name}: " + "; ".join(details))
        return "\n".join(lines)

    def format_rules_for_prompt(self) -> str:
        if not self.semantic_rules:
            return "- No domain semantic rules provided."

        lines = []
        for rule in self.semantic_rules:
            lines.append(f"- {rule.description}")
            if rule.sql_hint:
                lines.append(f"  SQL hint: {rule.sql_hint}")
        return "\n".join(lines)
