"""SQL generation agent backed by a local Ollama model."""

from __future__ import annotations

from local_agentic_analytics.agents.sql_cleaning import clean_sql_response
from local_agentic_analytics.prompts.sql_prompt import build_sql_prompt
from local_agentic_analytics.tools.ollama_tool import OllamaTool


class SQLAgent:
    """Generate DuckDB SQL from a user question and table schema."""

    def __init__(
        self,
        ollama_tool: OllamaTool | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if max_tokens < 1:
            raise ValueError("max_tokens must be greater than 0")

        self.ollama_tool = ollama_tool or OllamaTool.from_config()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_sql(self, question: str, schema: str) -> str:
        """Generate a SQL string only."""
        prompt = build_sql_prompt(question=question, schema=schema)
        response = self.ollama_tool.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        sql = clean_sql_response(response)

        if not sql:
            raise RuntimeError("SQL agent returned an empty response")

        return sql
