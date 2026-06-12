"""Domain adapter interface used by agents and workflows."""

from __future__ import annotations

from dataclasses import dataclass

from local_agentic_analytics.domain.profile import DatasetProfile


@dataclass(frozen=True)
class DomainAdapter:
    profile: DatasetProfile

    @property
    def table_name(self) -> str:
        return self.profile.table_name

    def format_sql_prompt_context(self) -> str:
        datetime_line = (
            f"Datetime column: {self.profile.datetime_column}"
            if self.profile.datetime_column
            else "Datetime column: not specified"
        )
        return (
            f"Dataset profile: {self.profile.display_name}\n"
            f"Dataset id: {self.profile.dataset_id}\n"
            f"Canonical table name: {self.profile.table_name}\n"
            f"{datetime_line}\n\n"
            "Domain columns:\n"
            f"{self.profile.format_columns_for_prompt()}\n\n"
            "Domain semantic rules:\n"
            f"{self.profile.format_rules_for_prompt()}"
        )
