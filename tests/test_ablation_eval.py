from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.sql_agent import SQLAgent
from local_agentic_analytics.agents.sql_cleaning import (
    clean_sql_response,
    extract_raw_model_sql,
)
from local_agentic_analytics.core.pipeline_toggles import PipelineToggles
from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.evaluation.ablation_eval import (
    ABLATION_CONFIGS,
    classify_route,
    run_ablation_evaluation,
    summarize_ablation_results,
)
from local_agentic_analytics.graph.workflow import SequentialAnalyticsWorkflow


TOTAL_ENERGY_QUESTION = (
    "Berapa total energi kWh pada tanggal 16 Desember 2006 "
    "berdasarkan Global_active_power?"
)
TOTAL_ENERGY_RESPONSE = (
    "```sql\n"
    "SELECT SUM(Global_active_power) AS total FROM electric_power "
    "WHERE CAST(datetime AS DATE) = DATE '2006-12-16';\n"
    "```"
)


class FakeDuckDBTool:
    def __init__(self, results_by_sql=None, errors_by_sql=None):
        self.results_by_sql = results_by_sql or {}
        self.errors_by_sql = errors_by_sql or {}
        self.executed_sql = []

    def get_schema(self, table_name: str) -> str:
        return f"{table_name}(datetime TIMESTAMP, Global_active_power DOUBLE)"

    def execute_query(self, sql: str) -> pd.DataFrame:
        self.executed_sql.append(sql)
        if sql in self.errors_by_sql:
            raise ValueError(self.errors_by_sql[sql])
        return self.results_by_sql[sql]


class FakeSQLAgent:
    def __init__(self, sql: str, raw_sql: str | None = None):
        self.sql = sql
        self.last_raw_generated_sql = raw_sql
        self.calls = []

    def generate_sql(self, question, schema, dataset_profile_context=None):
        self.calls.append(question)
        return self.sql


class RecordingRepairAgent:
    def __init__(self, repaired_sql: str = "SELECT 1"):
        self.repaired_sql = repaired_sql
        self.calls = []

    def repair_sql(
        self,
        failed_sql,
        error_message,
        schema,
        repair_attempted=False,
        user_question="",
    ):
        self.calls.append(failed_sql)
        return self.repaired_sql


class FakeReporterAgent:
    def __init__(self, answer: str = "Jawaban ringkas."):
        self.answer = answer

    def generate_answer(self, question, sql, query_result):
        return self.answer


class FakeOllamaTool:
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt, temperature, max_tokens):
        return self.response


def _tool_names(state: AnalyticsState) -> set[str]:
    return {event["tool"] for event in state.tool_calls}


# --- Toggle defaults reproduce current behavior --------------------------------


def test_pipeline_toggles_default_to_all_true():
    toggles = PipelineToggles()

    assert toggles.use_rule_based_resolver is True
    assert toggles.apply_domain_normalization is True
    assert toggles.use_semantic_guard is True
    assert toggles.use_repair is True


def test_workflow_defaults_to_all_true_toggles():
    workflow = SequentialAnalyticsWorkflow(
        duckdb_tool=FakeDuckDBTool(),
        sql_agent=FakeSQLAgent(sql="SELECT 1"),
        repair_agent=RecordingRepairAgent(),
        reporter_agent=FakeReporterAgent(),
    )

    assert workflow.toggles == PipelineToggles()


def test_default_toggles_run_resolver_guard_and_repair():
    failed_sql = "SELECT total_power FROM electric_power"
    repaired_sql = "SELECT SUM(Global_active_power) AS total FROM electric_power"
    duckdb_tool = FakeDuckDBTool(
        results_by_sql={repaired_sql: pd.DataFrame({"total": [100.0]})},
        errors_by_sql={failed_sql: "Invalid SQL query: column not found"},
    )
    repair_agent = RecordingRepairAgent(repaired_sql=repaired_sql)
    workflow = SequentialAnalyticsWorkflow(
        duckdb_tool=duckdb_tool,
        sql_agent=FakeSQLAgent(sql=failed_sql),
        repair_agent=repair_agent,
        reporter_agent=FakeReporterAgent(),
    )

    state = workflow.run("Berapa total konsumsi listrik?")

    assert state.success is True
    assert "rule_based_sql_resolver.resolve" in _tool_names(state)
    assert "sql_semantic_guard.validate" in _tool_names(state)
    assert len(repair_agent.calls) == 1


# --- Individual toggles turn scaffolding off -----------------------------------


def test_rule_based_disabled_forces_llm_path():
    sql = (
        "SELECT AVG(Global_active_power) AS avg_global_active_power_kw "
        "FROM electric_power "
        "WHERE CAST(datetime AS DATE) = DATE '2006-12-16'"
    )
    duckdb_tool = FakeDuckDBTool(results_by_sql={sql: pd.DataFrame({"avg": [1.23]})})
    sql_agent = FakeSQLAgent(sql=sql, raw_sql=sql)
    workflow = SequentialAnalyticsWorkflow(
        duckdb_tool=duckdb_tool,
        sql_agent=sql_agent,
        repair_agent=RecordingRepairAgent(),
        reporter_agent=FakeReporterAgent(),
        toggles=PipelineToggles(use_rule_based_resolver=False),
    )

    state = workflow.run(
        "Berapa rata-rata konsumsi daya aktif pada tanggal 16 Desember 2006?"
    )

    assert state.route == "llm_sql"
    assert sql_agent.calls  # LLM path was taken even though resolver would match
    assert "rule_based_sql_resolver.resolve" not in _tool_names(state)
    assert state.raw_generated_sql == sql


def test_semantic_guard_disabled_skips_validation_step():
    sql = "SELECT AVG(Global_active_power) AS avg_power FROM electric_power"
    duckdb_tool = FakeDuckDBTool(results_by_sql={sql: pd.DataFrame({"avg_power": [1.23]})})
    workflow = SequentialAnalyticsWorkflow(
        duckdb_tool=duckdb_tool,
        sql_agent=FakeSQLAgent(sql=sql, raw_sql=sql),
        repair_agent=RecordingRepairAgent(),
        reporter_agent=FakeReporterAgent(),
        toggles=PipelineToggles(
            use_rule_based_resolver=False,
            use_semantic_guard=False,
        ),
    )

    state = workflow.run("Berapa rata-rata konsumsi listrik?")

    assert state.success is True
    assert "sql_semantic_guard.validate" not in _tool_names(state)


def test_repair_disabled_propagates_execution_error():
    failed_sql = "SELECT total_power FROM electric_power"
    duckdb_tool = FakeDuckDBTool(
        errors_by_sql={failed_sql: "Invalid SQL query: column not found"},
    )
    repair_agent = RecordingRepairAgent()
    workflow = SequentialAnalyticsWorkflow(
        duckdb_tool=duckdb_tool,
        sql_agent=FakeSQLAgent(sql=failed_sql, raw_sql=failed_sql),
        repair_agent=repair_agent,
        reporter_agent=FakeReporterAgent(),
        toggles=PipelineToggles(
            use_rule_based_resolver=False,
            use_repair=False,
        ),
    )

    state = workflow.run("Berapa total konsumsi?")

    assert state.success is False
    assert state.repaired_sql is None
    assert repair_agent.calls == []
    assert "column not found" in (state.error_message or "")


# --- Domain normalization toggle and raw SQL capture ---------------------------


def test_clean_sql_response_skips_domain_normalization_but_keeps_fences():
    with_domain = clean_sql_response(
        TOTAL_ENERGY_RESPONSE,
        question=TOTAL_ENERGY_QUESTION,
        apply_domain_normalization=True,
    )
    without_domain = clean_sql_response(
        TOTAL_ENERGY_RESPONSE,
        question=TOTAL_ENERGY_QUESTION,
        apply_domain_normalization=False,
    )

    # Fence stripping and SELECT extraction happen in both modes.
    assert "```" not in without_domain
    assert without_domain.startswith("SELECT")
    # Domain rewriting (kWh conversion) only happens when enabled.
    assert "/ 60.0" in with_domain
    assert "/ 60" not in without_domain


def test_sql_agent_records_raw_model_sql_in_full_mode():
    agent = SQLAgent(
        ollama_tool=FakeOllamaTool(TOTAL_ENERGY_RESPONSE),
        apply_domain_normalization=True,
    )

    generated = agent.generate_sql(
        question=TOTAL_ENERGY_QUESTION,
        schema="electric_power(datetime TIMESTAMP, Global_active_power DOUBLE)",
    )

    assert "/ 60.0" in generated
    assert agent.last_raw_generated_sql == extract_raw_model_sql(TOTAL_ENERGY_RESPONSE)
    assert "/ 60" not in (agent.last_raw_generated_sql or "")
    assert (agent.last_raw_generated_sql or "").startswith("SELECT")


def test_workflow_populates_raw_generated_sql_on_llm_path():
    sql = "SELECT AVG(Global_active_power) AS avg_power FROM electric_power"
    raw_sql = "SELECT AVG(Global_active_power) AS avg FROM electric_power"
    duckdb_tool = FakeDuckDBTool(results_by_sql={sql: pd.DataFrame({"avg_power": [1.0]})})
    workflow = SequentialAnalyticsWorkflow(
        duckdb_tool=duckdb_tool,
        sql_agent=FakeSQLAgent(sql=sql, raw_sql=raw_sql),
        repair_agent=RecordingRepairAgent(),
        reporter_agent=FakeReporterAgent(),
        toggles=PipelineToggles(use_rule_based_resolver=False),
    )

    state = workflow.run("Berapa nilainya?")

    assert state.generated_sql == sql
    assert state.raw_generated_sql == raw_sql


# --- Ablation runner -----------------------------------------------------------


class FakeAblationWorkflow:
    def __init__(self, toggles: PipelineToggles):
        self.toggles = toggles

    def run(self, user_query: str) -> AnalyticsState:
        route = (
            "rule_based_sql"
            if self.toggles.use_rule_based_resolver
            else "llm_sql"
        )
        return AnalyticsState(
            user_query=user_query,
            generated_sql="SELECT 1 AS value",
            raw_generated_sql="SELECT 1 AS value",
            route=route,
            success=True,
        )


def test_run_ablation_evaluation_produces_expected_config_rows(tmp_path):
    gold_sql_path = tmp_path / "E001.sql"
    gold_sql_path.write_text("SELECT 1.0 AS value", encoding="utf-8")
    questions = [
        {
            "id": "E001",
            "question": "Berapa nilai?",
            "gold_sql_file": str(gold_sql_path),
            "expected_unit": "count",
        }
    ]
    duckdb_tool = FakeDuckDBTool(
        results_by_sql={
            "SELECT 1 AS value": pd.DataFrame({"value": [1.0]}),
            "SELECT 1.0 AS value": pd.DataFrame({"value": [1.0]}),
        }
    )

    rows = run_ablation_evaluation(
        questions,
        workflow_factory=FakeAblationWorkflow,
        duckdb_tool=duckdb_tool,
    )

    configs_in_order = [row["config"] for row in rows]
    assert configs_in_order == [config.name for config in ABLATION_CONFIGS]
    for row in rows:
        assert row["question_id"] == "E001"
        assert row["execution_success"] is True
        assert row["gold_success"] is True
        assert row["numeric_match"] is True

    summary = summarize_ablation_results(rows)
    assert set(summary["configs"]) == {config.name for config in ABLATION_CONFIGS}
    assert summary["configs"]["D_full"]["execution_success_rate"] == 1.0
    assert summary["configs"]["D_full"]["numeric_match_rate"] == 1.0

    route = summary["d_full_route_distribution"]
    assert route["rule_based_count"] == 1
    assert route["llm_count"] == 0
    assert route["rule_based_pct"] == 1.0


def test_classify_route_handles_repair_suffix():
    assert classify_route("rule_based_sql") == "rule_based"
    assert classify_route("rule_based_sql_with_repair") == "rule_based"
    assert classify_route("llm_sql") == "llm"
    assert classify_route("llm_sql_with_repair") == "llm"
    assert classify_route("") == "other"
