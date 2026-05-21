"""SQL repair agent backed by a local Ollama model."""

from __future__ import annotations

from local_agentic_analytics.agents.sql_cleaning import clean_sql_response
from local_agentic_analytics.prompts.repair_prompt import build_repair_prompt
from local_agentic_analytics.tools.ollama_tool import OllamaTool


class SQLRepairAgent:
    """Repair failed DuckDB SQL once for a query attempt."""

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

    def repair_sql(
        self,
        failed_sql: str,
        error_message: str,
        schema: str,
        repair_attempted: bool = False,
    ) -> str:
        """Repair a failed SQL query.

        ``repair_attempted`` is workflow state. Pass ``True`` to reject a second
        repair for the same user query.
        """
        if repair_attempted:
            raise RuntimeError("SQL repair already attempted for this query")

        prompt = build_repair_prompt(
            failed_sql=failed_sql,
            error_message=error_message,
            schema=schema,
        )
        response = self.ollama_tool.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        sql = clean_sql_response(response)

        if not sql:
            raise RuntimeError("SQL repair agent returned an empty response")

        return sql

RepairAgent = SQLRepairAgent
