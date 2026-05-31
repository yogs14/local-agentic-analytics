"""Shared state objects for sequential analytics workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalyticsState:
    """Minimal state passed through the local analytics workflow."""

    user_query: str
    schema: str | None = None
    generated_sql: str | None = None
    repaired_sql: str | None = None
    sql_result: dict[str, Any] | None = None
    error_message: str | None = None
    final_answer: str | None = None
    latency: dict[str, float] = field(default_factory=dict)
    selected_tools: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieved_context: list[dict[str, Any]] = field(default_factory=list)
    route: str | None = None
    success: bool = False
