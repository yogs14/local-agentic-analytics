"""Reporter agent for concise Indonesian analytics answers."""

from __future__ import annotations

from typing import Any

from local_agentic_analytics.prompts.reporter_prompt import build_reporter_prompt
from local_agentic_analytics.tools.ollama_tool import OllamaTool


class ReporterAgent:
    """Generate Indonesian answers from executed SQL and compact query results."""

    def __init__(
        self,
        ollama_tool: OllamaTool | None = None,
        temperature: float = 0.1,
        max_tokens: int = 192,
    ):
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if max_tokens < 1:
            raise ValueError("max_tokens must be greater than 0")

        self.ollama_tool = ollama_tool or OllamaTool.from_config()
        self.temperature = temperature
        # Reporter output should stay concise but needs room for a short answer.
        self.max_tokens = max_tokens

    def generate_answer(self, question: str, sql: str, query_result: Any) -> str:
        """Generate a concise Indonesian answer from query context."""
        prompt = build_reporter_prompt(
            question=question,
            sql=sql,
            query_result=query_result,
        )
        response = self.ollama_tool.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        answer = response.strip()

        if not answer:
            raise RuntimeError("Reporter agent returned an empty response")

        return answer
