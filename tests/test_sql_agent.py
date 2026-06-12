from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.sql_agent import SQLAgent
from local_agentic_analytics.prompts.sql_prompt import build_sql_prompt


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


def test_build_sql_prompt_contains_schema_question_and_rules():
    prompt = build_sql_prompt(
        question="Total energy per day?",
        schema="electric_power(datetime TIMESTAMP, Global_active_power DOUBLE)",
    )

    assert "DuckDB SQL" in prompt
    assert "Return SQL only" in prompt
    assert "Do not invent generic table names" in prompt
    assert "CAST(datetime AS DATE)" in prompt
    assert "SUM(Global_active_power) / 60.0 AS total_energy_kwh" in prompt
    assert "COUNT(*) FILTER (WHERE <column> IS NULL)" in prompt
    assert "COUNT(DISTINCT CASE WHEN" in prompt
    assert "MAX(Global_active_power) AS max_global_active_power_kw" in prompt
    assert "COUNT(*) AS record_count" in prompt
    assert "electric_power" in prompt
    assert "Total energy per day?" in prompt


def test_build_sql_prompt_accepts_custom_domain_context():
    prompt = build_sql_prompt(
        question="Total sales?",
        schema="sales(amount DOUBLE)",
        domain_context="Domain semantic rules:\n- Use SUM(amount) for total sales.",
    )

    assert "Use SUM(amount) for total sales" in prompt
    assert "Dataset profile and semantic rules" in prompt


def test_sql_agent_returns_sql_string_from_ollama_response():
    fake_tool = FakeOllamaTool(
        "SELECT DATE_TRUNC('day', datetime) AS day, "
        "SUM(Global_active_power) AS total_power FROM electric_power GROUP BY day"
    )
    agent = SQLAgent(ollama_tool=fake_tool, temperature=0.0, max_tokens=256)

    sql = agent.generate_sql(
        question="Total power by day",
        schema="electric_power(datetime TIMESTAMP, Global_active_power DOUBLE)",
    )

    assert sql.startswith("SELECT")
    assert "GROUP BY day" in sql
    assert fake_tool.calls[0]["temperature"] == 0.0
    assert fake_tool.calls[0]["max_tokens"] == 256


def test_sql_agent_strips_code_fence_if_model_returns_markdown():
    fake_tool = FakeOllamaTool("```sql\nSELECT COUNT(*) FROM electric_power;\n```")
    agent = SQLAgent(ollama_tool=fake_tool)

    sql = agent.generate_sql(
        question="How many rows?",
        schema="electric_power(datetime TIMESTAMP)",
    )

    assert sql == "SELECT COUNT(*) FROM electric_power;"
    assert fake_tool.calls[0]["max_tokens"] == 96


def test_sql_agent_keeps_only_first_sql_statement():
    fake_tool = FakeOllamaTool("SELECT COUNT(*) FROM electric_power; SELECT 2;")
    agent = SQLAgent(ollama_tool=fake_tool)

    sql = agent.generate_sql(
        question="How many rows?",
        schema="electric_power(datetime TIMESTAMP)",
    )

    assert sql == "SELECT COUNT(*) FROM electric_power;"
    assert fake_tool.calls[0]["max_tokens"] == 96


def test_sql_agent_can_print_prompt_length_when_debug_enabled(capsys):
    fake_tool = FakeOllamaTool("SELECT 1;")
    agent = SQLAgent(ollama_tool=fake_tool, debug_prompt_length=True)

    sql = agent.generate_sql(
        question="Berapa jumlah record?",
        schema="Table: electric_power\nColumns:\ndatetime: TIMESTAMP",
    )

    assert sql == "SELECT COUNT(*) AS record_count FROM electric_power;"
    assert "SQL prompt length:" in capsys.readouterr().out


def test_sql_agent_normalizes_timestamp_date_filter():
    fake_tool = FakeOllamaTool(
        "SELECT AVG(Global_active_power) FROM electric_power "
        "WHERE datetime = CAST(datetime AS DATE) = DATE '2006-12-16'"
    )
    agent = SQLAgent(ollama_tool=fake_tool)

    sql = agent.generate_sql(
        question="Rata-rata pada 16 Desember 2006",
        schema="Table: electric_power\nColumns:\ndatetime: TIMESTAMP",
    )

    assert (
        sql
        == "SELECT AVG(Global_active_power) FROM electric_power "
        "WHERE CAST(datetime AS DATE) = DATE '2006-12-16'"
    )


def test_sql_agent_normalizes_total_energy_kwh_conversion():
    fake_tool = FakeOllamaTool(
        "SELECT SUM(Global_active_power) FROM electric_power "
        "WHERE CAST(datetime AS DATE) = DATE '2006-12-16'"
    )
    agent = SQLAgent(ollama_tool=fake_tool)

    sql = agent.generate_sql(
        question="Berapa total energi kWh pada tanggal 16 Desember 2006 berdasarkan Global_active_power?",
        schema="Table: electric_power\nColumns:\nGlobal_active_power: DOUBLE",
    )

    assert "SUM(Global_active_power) / 60.0 AS total_energy_kwh" in sql
    assert "CAST(datetime AS DATE) = DATE '2006-12-16'" in sql


def test_sql_agent_normalizes_missing_value_count():
    fake_tool = FakeOllamaTool(
        "SELECT COUNT(DISTINCT CASE WHEN Global_active_power IS NULL THEN 1 END) "
        "FROM electric_power;"
    )
    agent = SQLAgent(ollama_tool=fake_tool)

    sql = agent.generate_sql(
        question="Berapa jumlah missing value pada kolom Global_active_power?",
        schema="Table: electric_power\nColumns:\nGlobal_active_power: DOUBLE",
    )

    assert (
        sql
        == "SELECT COUNT(*) FILTER (WHERE Global_active_power IS NULL) "
        "AS missing_global_active_power_count FROM electric_power;"
    )


def test_sql_agent_normalizes_record_count_date_filter():
    fake_tool = FakeOllamaTool(
        "SELECT COUNT(*) FILTER (WHERE CAST(datetime AS DATE) = DATE '2006-12-16') "
        "AS missing_<column>_count FROM electric_power;"
    )
    agent = SQLAgent(ollama_tool=fake_tool)

    sql = agent.generate_sql(
        question="Berapa jumlah record pada tanggal 16 Desember 2006?",
        schema="Table: electric_power\nColumns:\ndatetime: TIMESTAMP",
    )

    assert (
        sql
        == "SELECT COUNT(*) AS record_count FROM electric_power "
        "WHERE CAST(datetime AS DATE) = DATE '2006-12-16';"
    )


def test_sql_agent_adds_energy_unit_aliases():
    fake_tool = FakeOllamaTool(
        "SELECT AVG(Voltage) FROM electric_power "
        "WHERE CAST(datetime AS DATE) = DATE '2006-12-16'"
    )
    agent = SQLAgent(ollama_tool=fake_tool)

    sql = agent.generate_sql(
        question="Berapa tegangan rata-rata pada tanggal 16 Desember 2006?",
        schema="Table: electric_power\nColumns:\nVoltage: DOUBLE",
    )

    assert "AVG(Voltage) AS avg_voltage_v" in sql


def test_sql_agent_normalizes_max_active_power_value():
    fake_tool = FakeOllamaTool(
        "SELECT datetime, Global_active_power AS max_global_active_power_kw "
        "FROM electric_power WHERE CAST(datetime AS DATE) = DATE '2006-12-16' "
        "ORDER BY Global_active_power DESC LIMIT 1"
    )
    agent = SQLAgent(ollama_tool=fake_tool)

    sql = agent.generate_sql(
        question="Berapa daya aktif maksimum pada tanggal 16 Desember 2006?",
        schema="Table: electric_power\nColumns:\nGlobal_active_power: DOUBLE",
    )

    assert (
        sql
        == "SELECT MAX(Global_active_power) AS max_global_active_power_kw "
        "FROM electric_power WHERE CAST(datetime AS DATE) = DATE '2006-12-16';"
    )


def test_sql_agent_rejects_empty_question():
    fake_tool = FakeOllamaTool("SELECT 1")
    agent = SQLAgent(ollama_tool=fake_tool)

    with pytest.raises(ValueError, match="question must not be empty"):
        agent.generate_sql(question="", schema="electric_power(id INTEGER)")
