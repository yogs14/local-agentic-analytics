"""SQL generation agent backed by a local Ollama model."""

from __future__ import annotations

from local_agentic_analytics.agents.sql_cleaning import (
    clean_sql_response,
    extract_raw_model_sql,
)
from local_agentic_analytics.domain.adapters import DomainAdapter
from local_agentic_analytics.prompts.sql_prompt import build_sql_prompt
from local_agentic_analytics.tools.ollama_tool import OllamaTool


class SQLAgent:
    """Generate DuckDB SQL from a user question and table schema."""

    def __init__(
        self,
        ollama_tool: OllamaTool | None = None,
        temperature: float = 0.0,
        max_tokens: int = 96,
        domain_adapter: DomainAdapter | None = None,
        dataset_profile_context: str | None = None,
        debug_prompt_length: bool = False,
        apply_domain_normalization: bool = True,
    ):
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if max_tokens < 1:
            raise ValueError("max_tokens must be greater than 0")

        self.ollama_tool = ollama_tool or OllamaTool.from_config()
        self.temperature = temperature
        # SQL generation stays short because the expected output is one SQL query.
        self.max_tokens = max_tokens
        self.domain_adapter = domain_adapter
        self.dataset_profile_context = dataset_profile_context
        self.debug_prompt_length = debug_prompt_length
        self.apply_domain_normalization = apply_domain_normalization
        # Raw model SQL (fence-strip + common normalization, no domain rewrite)
        # from the most recent generation, for faithful ablation analysis.
        self.last_raw_generated_sql: str | None = None

    def generate_sql(
        self,
        question: str,
        schema: str,
        dataset_profile_context: str | None = None,
    ) -> str:
        """Generate a SQL string only."""
        profile_context = dataset_profile_context or self.dataset_profile_context

        if self.domain_adapter is not None:
            adapter_context = self.domain_adapter.format_sql_prompt_context()
            profile_context = (
                f"{profile_context}\n\n{adapter_context}"
                if profile_context
                else adapter_context
            )

        prompt = build_sql_prompt(
            question=question,
            schema=schema,
            dataset_profile_context=profile_context,
        )
        if self.debug_prompt_length:
            print(f"SQL prompt length: {len(prompt)} characters")

        response = self.ollama_tool.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.last_raw_generated_sql = extract_raw_model_sql(response)
        sql = clean_sql_response(
            response,
            question=question,
            apply_domain_normalization=self.apply_domain_normalization,
        )

        if not sql:
            raise RuntimeError("SQL agent returned an empty response")

        return sql
