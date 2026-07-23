"""Insight agent for explaining chart metadata and compact statistics."""

from __future__ import annotations

import re
from typing import Any, Callable

from local_agentic_analytics.prompts.insight_prompt import (
    build_energy_finetune_prompt,
    build_insight_prompt,
)
from local_agentic_analytics.tools.ollama_tool import OllamaTool


# ATX headings at the start of a line, and bold/italic ``*``/``**`` emphasis. The
# narrator's system prompt forbids Markdown, but small models still leak it; left
# unstripped it renders literally (``##``, ``**``) in the LaTeX/PDF report.
# Underscores are deliberately preserved so column identifiers such as
# ``Sub_metering_3`` and ``Global_active_power`` survive intact.
_MARKDOWN_HEADING_RE = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]*")
_MARKDOWN_EMPHASIS_RE = re.compile(r"\*+")


def sanitize_narrative(text: str) -> str:
    """Remove leaked Markdown headings/emphasis while preserving prose and numbers."""
    without_headings = _MARKDOWN_HEADING_RE.sub("", text)
    without_emphasis = _MARKDOWN_EMPHASIS_RE.sub("", without_headings)
    # Collapse blank lines created by removing standalone heading lines.
    lines = [line.rstrip() for line in without_emphasis.splitlines()]
    collapsed = "\n".join(line for line in lines if line.strip())
    return collapsed.strip()


class InsightAgent:
    """Generate short Indonesian narratives from chart stats."""

    def __init__(
        self,
        ollama_tool: OllamaTool | None = None,
        temperature: float = 0.1,
        max_tokens: int = 384,
        prompt_builder: Callable[[dict[str, Any]], str] = build_insight_prompt,
    ):
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if max_tokens < 1:
            raise ValueError("max_tokens must be greater than 0")

        self.ollama_tool = ollama_tool or OllamaTool.from_config()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_builder = prompt_builder

    @classmethod
    def for_energy_finetune(
        cls,
        ollama_tool: OllamaTool | None = None,
        temperature: float = 0.4,
        max_tokens: int = 384,
    ) -> "InsightAgent":
        """Build an InsightAgent backed by the energy fine-tune model.

        Reads the ``insight_model`` section of ``model.yaml`` (defaulting to
        ``gemma2-energy-insight`` via ``OLLAMA_INSIGHT_MODEL``) and feeds it the
        flat stats-block format it was trained on. ``temperature`` defaults to
        ``0.4`` to match the fine-tune's training setting.
        """
        tool = ollama_tool or OllamaTool.from_config(section="insight_model")
        return cls(
            ollama_tool=tool,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_builder=build_energy_finetune_prompt,
        )

    def generate_insight(self, chart_context: dict[str, Any]) -> str:
        prompt = self.prompt_builder(chart_context)
        response = self.ollama_tool.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        insight = sanitize_narrative(response.strip())
        if not insight:
            raise RuntimeError("Insight agent returned an empty response")

        return insight
