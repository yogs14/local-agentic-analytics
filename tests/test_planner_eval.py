from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.route_types import RouteDecision
from local_agentic_analytics.evaluation.planner_eval import (
    PLANNER_EVAL_COLUMNS,
    PlannerConfig,
    _NullDuckDBTool,
    load_planner_questions,
    run_planner_evaluation,
    summarize_planner_results,
    write_planner_results,
)
from local_agentic_analytics.graph.workflow import SequentialAnalyticsWorkflow


class FakeMetricsOllamaTool:
    def get_last_metrics(self):
        return {}


class FakePlannerAgent:
    """Deterministic planner that always returns STRUCTURED_SQL."""

    def __init__(self):
        self.calls = []
        self.ollama_tool = FakeMetricsOllamaTool()

    def plan_route(self, question):
        self.calls.append(question)
        return RouteDecision.STRUCTURED_SQL


class NoopAuditLogger:
    def log(self, **kwargs):
        return None


def _factory(config: PlannerConfig) -> SequentialAnalyticsWorkflow:
    return SequentialAnalyticsWorkflow(
        domain="finance",
        toggles=config.toggles,
        duckdb_tool=_NullDuckDBTool(),
        planner_agent=FakePlannerAgent(),
        audit_logger=NoopAuditLogger(),
    )


def test_planner_evaluation_scores_all_routes_correctly():
    questions = load_planner_questions()
    rows = run_planner_evaluation(questions, workflow_factory=_factory)
    summary = summarize_planner_results(rows)

    assert set(summary["configs"]) == {"rule_based_only", "rule_based_plus_llm"}
    for stats in summary["configs"].values():
        assert stats["total"] == 15
        # The deterministic resolver + STRUCTURED_SQL fake route everything right.
        assert stats["correct"] == 15
        assert stats["accuracy"] == 1.0
        # Confusion is purely diagonal when accuracy is perfect.
        for expected, predicted_counts in stats["confusion"].items():
            assert set(predicted_counts) == {expected}


def test_planner_evaluation_breakdown_shows_llm_contribution():
    questions = load_planner_questions()
    rows = run_planner_evaluation(questions, workflow_factory=_factory)
    summary = summarize_planner_results(rows)

    rule_based = summary["configs"]["rule_based_only"]["route_source_breakdown"]
    plus_llm = summary["configs"]["rule_based_plus_llm"]["route_source_breakdown"]

    # Without the LLM, every decision is made by the deterministic resolver.
    assert rule_based == {"rule_based": 15}
    # With the LLM enabled, the 8 ambiguous SQL questions are decided by the LLM,
    # while the 7 explicit news/hybrid questions stay rule-based.
    assert plus_llm == {"rule_based": 7, "llm": 8}


def test_write_planner_results_writes_all_columns(tmp_path):
    questions = load_planner_questions()
    rows = run_planner_evaluation(questions, workflow_factory=_factory)
    csv_path = tmp_path / "planner_eval.csv"

    write_planner_results(rows, csv_path)

    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(PLANNER_EVAL_COLUMNS)
    assert len(lines) == 1 + len(rows)
