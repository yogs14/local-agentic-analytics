from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.prompting_comparison import (
    STRATEGIES,
    STRATEGY_DECOMPOSED,
    STRATEGY_FEW_SHOT,
    STRATEGY_ZERO_SHOT,
    build_few_shot_prompt,
    build_zero_shot_prompt,
    generate_with_strategy,
    run_prompting_comparison,
)


SCHEMA = "Table electric_power(datetime TIMESTAMP, Global_active_power DOUBLE)"


class FakeOllama:
    """Returns queued responses and canned generation metrics."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, temperature=0.0, max_tokens=256):
        assert temperature == 0.0
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("no more responses")
        return self.responses.pop(0)

    def get_last_metrics(self):
        return {"eval_count": 40, "eval_duration": 2.0}


class FakeDuckDB:
    def __init__(self, frames):
        self.frames = frames

    def execute_query(self, sql):
        result = self.frames[sql]
        if isinstance(result, Exception):
            raise result
        return result


def test_prompt_builders_include_schema_and_question():
    zero = build_zero_shot_prompt("Berapa X?", SCHEMA)
    assert SCHEMA in zero
    assert "Berapa X?" in zero
    assert "Example" not in zero

    few = build_few_shot_prompt("Berapa X?", SCHEMA, "energy")
    assert "Examples:" in few
    assert "BETWEEN DATE" in few  # date-range exemplar is present
    assert few.index("Examples:") < few.index("Q: Berapa X?")


def test_generate_zero_shot_single_call():
    tool = FakeOllama(["SELECT 1 AS v;"])

    result = generate_with_strategy(
        STRATEGY_ZERO_SHOT, "Q?", SCHEMA, tool
    )

    assert result.sql.startswith("SELECT 1")
    assert result.n_llm_calls == 1
    assert result.tokens_per_second == pytest.approx(20.0)
    assert result.error == ""


def test_generate_few_shot_strips_markdown():
    tool = FakeOllama(["```sql\nSELECT 2 AS v;\n```"])

    result = generate_with_strategy(
        STRATEGY_FEW_SHOT, "Q?", SCHEMA, tool
    )

    assert "```" not in result.sql
    assert "SELECT 2" in result.sql


def test_generate_decomposed_three_calls():
    tool = FakeOllama(
        [
            "- table: electric_power\n- columns: Global_active_power",
            "SELECT AVG(Global_active_power) FROM electric_power;",
            "SELECT AVG(Global_active_power) AS avg_kw FROM electric_power;",
        ]
    )

    result = generate_with_strategy(
        STRATEGY_DECOMPOSED, "Q?", SCHEMA, tool
    )

    assert result.n_llm_calls == 3
    assert "avg_kw" in result.sql
    # Draft feeds the refine prompt.
    assert "SELECT AVG(Global_active_power) FROM electric_power;" in tool.prompts[2]


def test_generate_decomposed_falls_back_to_draft():
    tool = FakeOllama(
        [
            "links",
            "SELECT 1 AS draft;",
            "Maaf, tidak ada SQL di sini.",
        ]
    )

    result = generate_with_strategy(
        STRATEGY_DECOMPOSED, "Q?", SCHEMA, tool
    )

    assert "draft" in result.sql


def test_generate_records_error():
    tool = FakeOllama([])  # immediately raises

    result = generate_with_strategy(STRATEGY_ZERO_SHOT, "Q?", SCHEMA, tool)

    assert result.sql == ""
    assert "no more responses" in result.error


def test_generate_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        generate_with_strategy("nope", "Q?", SCHEMA, FakeOllama([]))


def test_run_prompting_comparison_rows(tmp_path):
    gold_sql = "SELECT 1.0 AS value"
    gold_path = tmp_path / "G1.sql"
    gold_path.write_text(gold_sql, encoding="utf-8")
    questions = [
        {
            "id": "E101",
            "question": "Berapa nilai?",
            "gold_sql_file": str(gold_path),
            "expected_unit": "kW",
        }
    ]
    agent_sql = "SELECT 1 AS value;"
    duckdb_tool = FakeDuckDB(
        {
            agent_sql: pd.DataFrame({"value": [1.0]}),
            gold_sql: pd.DataFrame({"value": [1.0]}),
        }
    )
    tool = FakeOllama([agent_sql, agent_sql])

    rows = run_prompting_comparison(
        questions,
        (STRATEGY_ZERO_SHOT, STRATEGY_FEW_SHOT),
        tool,
        duckdb_tool,
        schema_context=SCHEMA,
        logger=lambda message: None,
    )

    assert [row["config"] for row in rows] == [
        STRATEGY_ZERO_SHOT,
        STRATEGY_FEW_SHOT,
    ]
    for row in rows:
        assert row["execution_success"] is True
        assert row["numeric_match"] is True
        assert row["result_match_full"] is True
        assert row["n_llm_calls"] == 1
        assert row["latency_total"] > 0


def test_strategies_constant():
    assert STRATEGIES == ("zero_shot", "few_shot_static", "decomposed")
