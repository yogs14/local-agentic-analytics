"""Retrieval route targets selected by the planner.

The planner moves one real decision to the agent: which retrieval route the next
node should pursue. Each member is a stable string label so it can be parsed out
of an SLM response, logged in the audit trail, and compared against the
``expected_source`` gold labels in ``data/evaluation/finance_questions.json``.
"""

from __future__ import annotations

from enum import Enum


class RouteDecision(str, Enum):
    """Possible retrieval routes for a user question."""

    STRUCTURED_SQL = "STRUCTURED_SQL"
    RAG_NEWS = "RAG_NEWS"
    HYBRID = "HYBRID"

    def __str__(self) -> str:  # pragma: no cover - trivial.
        return self.value

    @classmethod
    def from_label(cls, label: str | None) -> "RouteDecision | None":
        """Return the route matching ``label`` (case-insensitive), or None."""
        if not label:
            return None
        normalized = label.strip().upper()
        for route in cls:
            if route.value == normalized:
                return route
        return None
