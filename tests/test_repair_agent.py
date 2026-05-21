from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.repair_agent import SQLRepairAgent
from local_agentic_analytics.prompts.repair_prompt import build_repair_prompt


class FakeOllamaTool:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 512) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self.response


def test_build_repair_prompt_contains_schema_failed_sql_error_and_rules():
    prompt = build_repair_prompt(
        failed_sql="SELECT total_power FROM electric_power",
        error_message="Binder Error: Referenced column total_power not found",
        schema="electric_power(Global_active_power DOUBLE)",
    )

    assert "DuckDB SQL repair" in prompt
    assert "Return SQL only" in prompt
    assert "Do not add catalog or schema prefixes" in prompt
    assert "CAST(column AS DATE)" in prompt
    assert "Global_active_power" in prompt
    assert "total_power" in prompt
    assert "Referenced column" in prompt


def test_repair_agent_returns_corrected_sql_string():
    fake_tool = FakeOllamaTool(
        "SELECT SUM(Global_active_power) AS total_power FROM electric_power;"
    )
    agent = SQLRepairAgent(ollama_tool=fake_tool, temperature=0.0, max_tokens=256)

    sql = agent.repair_sql(
        failed_sql="SELECT total_power FROM electric_power",
        error_message="Binder Error: Referenced column total_power not found",
        schema="electric_power(Global_active_power DOUBLE)",
    )

    assert sql == "SELECT SUM(Global_active_power) AS total_power FROM electric_power;"
    assert fake_tool.calls[0]["temperature"] == 0.0
    assert fake_tool.calls[0]["max_tokens"] == 256


def test_repair_agent_strips_code_fence_if_model_returns_markdown():
    fake_tool = FakeOllamaTool("```sql\nSELECT COUNT(*) FROM electric_power;\n```")
    agent = SQLRepairAgent(ollama_tool=fake_tool)

    sql = agent.repair_sql(
        failed_sql="SELECT COUNT(id) FROM electric_power",
        error_message="Binder Error: Referenced column id not found",
        schema="electric_power(datetime TIMESTAMP)",
    )

    assert sql == "SELECT COUNT(*) FROM electric_power;"


def test_repair_agent_normalizes_timestamp_date_filter():
    fake_tool = FakeOllamaTool(
        "SELECT AVG(Global_active_power) FROM electric_power "
        "WHERE datetime = DATE '2006-12-16'"
    )
    agent = SQLRepairAgent(ollama_tool=fake_tool)

    sql = agent.repair_sql(
        failed_sql="SELECT AVG(Global_active_power) FROM electric_power WHERE datetime = DATE '2006-12-16'",
        error_message="Result returned NULL for a whole-day timestamp filter",
        schema="Table: electric_power\nColumns:\ndatetime: TIMESTAMP",
    )

    assert (
        sql
        == "SELECT AVG(Global_active_power) FROM electric_power "
        "WHERE CAST(datetime AS DATE) = DATE '2006-12-16'"
    )


def test_repair_agent_rejects_second_repair_attempt():
    fake_tool = FakeOllamaTool("SELECT 1")
    agent = SQLRepairAgent(ollama_tool=fake_tool)

    with pytest.raises(RuntimeError, match="already attempted"):
        agent.repair_sql(
            failed_sql="SELECT missing FROM electric_power",
            error_message="Binder Error",
            schema="electric_power(datetime TIMESTAMP)",
            repair_attempted=True,
        )


def test_repair_agent_rejects_empty_error_message():
    fake_tool = FakeOllamaTool("SELECT 1")
    agent = SQLRepairAgent(ollama_tool=fake_tool)

    with pytest.raises(ValueError, match="error_message must not be empty"):
        agent.repair_sql(
            failed_sql="SELECT missing FROM electric_power",
            error_message="",
            schema="electric_power(datetime TIMESTAMP)",
        )
