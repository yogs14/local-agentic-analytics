"""Insight agent for explaining chart metadata and compact statistics."""

from __future__ import annotations

from typing import Any

from local_agentic_analytics.prompts.insight_prompt import build_insight_prompt
from local_agentic_analytics.tools.ollama_tool import OllamaTool


class InsightAgent:
    """Generate short Indonesian narratives from chart stats."""

    def __init__(
        self,
        ollama_tool: OllamaTool | None = None,
        temperature: float = 0.1,
        max_tokens: int = 384,
    ):
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if max_tokens < 1:
            raise ValueError("max_tokens must be greater than 0")

        self.ollama_tool = ollama_tool or OllamaTool.from_config()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_insight(self, chart_context: dict[str, Any]) -> str:
        prompt = build_insight_prompt(chart_context)
        response = self.ollama_tool.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        insight = response.strip()
        if not insight:
            raise RuntimeError("Insight agent returned an empty response")

        return insight
