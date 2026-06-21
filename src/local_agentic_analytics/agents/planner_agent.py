"""LLM route planner backed by the same local Ollama model as the SQL agent.

This is the LLM half of the planner: it is consulted only when the deterministic
``RuleBasedRouteResolver`` is unsure. Because gemma2:2b is small and may ramble,
parsing is deliberately tolerant and always falls back to STRUCTURED_SQL rather
than raising, so the planner can never break the workflow.
"""

from __future__ import annotations

import re

from local_agentic_analytics.agents.route_types import RouteDecision
from local_agentic_analytics.prompts.planner_prompt import build_planner_prompt
from local_agentic_analytics.tools.ollama_tool import OllamaTool


# Match the first standalone route label anywhere in the (uppercased) response.
_LABEL_PATTERN = re.compile(r"\b(STRUCTURED_SQL|RAG_NEWS|HYBRID)\b")


class PlannerAgent:
    """Pick a retrieval route for a question using the local SLM."""

    def __init__(
        self,
        ollama_tool: OllamaTool | None = None,
        temperature: float = 0.0,
        max_tokens: int = 16,
    ):
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if max_tokens < 1:
            raise ValueError("max_tokens must be greater than 0")

        self.ollama_tool = ollama_tool or OllamaTool.from_config()
        self.temperature = temperature
        # The expected output is a single short label, so keep generation tiny.
        self.max_tokens = max_tokens
        self.last_raw_response: str | None = None

    def plan_route(self, question: str) -> RouteDecision:
        """Return the route the SLM selects, defaulting to STRUCTURED_SQL."""
        prompt = build_planner_prompt(question)
        response = self.ollama_tool.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.last_raw_response = response
        return parse_route_label(response)

    def get_last_metrics(self) -> dict[str, float | int]:
        """Return profiling metrics from the most recent planning call."""
        get_last_metrics = getattr(self.ollama_tool, "get_last_metrics", None)
        if not callable(get_last_metrics):
            return {}
        metrics = get_last_metrics()
        return dict(metrics) if isinstance(metrics, dict) else {}


def parse_route_label(response: str | None) -> RouteDecision:
    """Tolerantly extract a route label, defaulting to STRUCTURED_SQL.

    The model output is uppercased and scanned for the first valid label token,
    so prose like ``"Rutenya adalah RAG_NEWS karena ..."`` still resolves. Any
    unparseable or empty response degrades safely to STRUCTURED_SQL.
    """
    if not response:
        return RouteDecision.STRUCTURED_SQL

    match = _LABEL_PATTERN.search(response.upper())
    if not match:
        return RouteDecision.STRUCTURED_SQL

    route = RouteDecision.from_label(match.group(1))
    return route or RouteDecision.STRUCTURED_SQL
