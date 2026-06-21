"""Planner routing evaluation: measure how well routes are selected.

Runs the planner over the finance gold question set, maps each question's
``expected_source`` (sql/rag/hybrid) to a :class:`RouteDecision`, and reports
routing accuracy, a per-route confusion breakdown, and the distribution of
decision sources (rule_based / llm / forced_energy / default). It compares at
least two configurations -- rule-based-only versus rule-based+LLM -- so the
contribution of the LLM planner is measurable, mirroring the ablation harness.

This is a measurement tool: it must not tune the planner it evaluates.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from local_agentic_analytics.agents.route_types import RouteDecision
from local_agentic_analytics.core.config import PROJECT_ROOT
from local_agentic_analytics.core.pipeline_toggles import PipelineToggles
from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.graph.workflow import SequentialAnalyticsWorkflow


DEFAULT_QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "finance_questions.json"
)
DEFAULT_CSV_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "planner_eval.csv"
)
DEFAULT_SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "planner_eval_summary.json"
)

PLANNER_EVAL_COLUMNS = (
    "config",
    "question_id",
    "question",
    "expected_source",
    "expected_route",
    "predicted_route",
    "route_source",
    "route_reasoning",
    "correct",
    "error_message",
)

# Map the gold ``expected_source`` label to the route the planner should choose.
EXPECTED_ROUTE_BY_SOURCE = {
    "sql": RouteDecision.STRUCTURED_SQL,
    "rag": RouteDecision.RAG_NEWS,
    "hybrid": RouteDecision.HYBRID,
}

ALL_ROUTES = tuple(route.value for route in RouteDecision)


@dataclass(frozen=True)
class PlannerConfig:
    """One named planner configuration to evaluate."""

    name: str
    toggles: PipelineToggles


PLANNER_CONFIGS: tuple[PlannerConfig, ...] = (
    PlannerConfig(
        name="rule_based_only",
        toggles=PipelineToggles(use_planner=False),
    ),
    PlannerConfig(
        name="rule_based_plus_llm",
        toggles=PipelineToggles(use_planner=True),
    ),
)


class _NullDuckDBTool:
    """Stand-in DuckDB tool so workflow construction never opens a database.

    Route planning never touches DuckDB, so this keeps the evaluator free of
    database and model dependencies while reusing the real decision logic.
    """

    def get_schema(self, table_name: str) -> str:  # pragma: no cover - unused.
        return ""

    def execute_query(self, sql: str):  # pragma: no cover - unused.
        raise RuntimeError("DuckDB is not used during planner evaluation")


class _NoopAuditLogger:
    def log(self, **kwargs: Any) -> None:
        return None


WorkflowFactory = Callable[[PlannerConfig], Any]


def _default_workflow_factory(config: PlannerConfig) -> SequentialAnalyticsWorkflow:
    return SequentialAnalyticsWorkflow(
        domain="finance",
        toggles=config.toggles,
        duckdb_tool=_NullDuckDBTool(),
        audit_logger=_NoopAuditLogger(),
    )


def run_planner_evaluation(
    questions: list[dict[str, Any]],
    configs: tuple[PlannerConfig, ...] = PLANNER_CONFIGS,
    workflow_factory: WorkflowFactory | None = None,
) -> list[dict[str, Any]]:
    """Run every config against every question and return flat result rows."""
    workflow_factory = workflow_factory or _default_workflow_factory

    rows: list[dict[str, Any]] = []
    for config in configs:
        workflow = workflow_factory(config)
        for question in questions:
            rows.append(
                run_single_planner_question(
                    question=question,
                    config=config,
                    workflow=workflow,
                )
            )
    return rows


def run_single_planner_question(
    question: dict[str, Any],
    config: PlannerConfig,
    workflow: Any,
) -> dict[str, Any]:
    """Plan the route for one question and compare it to the gold source."""
    question_id = str(question.get("id", ""))
    question_text = str(question.get("question", ""))
    expected_source = str(question.get("expected_source", "")).lower()
    expected_route = EXPECTED_ROUTE_BY_SOURCE.get(expected_source)

    state = AnalyticsState(user_query=question_text)
    predicted_route = ""
    route_source = ""
    route_reasoning = ""
    error_message = ""
    try:
        route = workflow._plan_route(state)
        predicted_route = route.value
        route_source = state.route_source or ""
        route_reasoning = state.route_reasoning or ""
    except Exception as exc:  # pragma: no cover - planner never raises.
        error_message = str(exc)

    correct = bool(expected_route) and predicted_route == expected_route.value

    return {
        "config": config.name,
        "question_id": question_id,
        "question": question_text,
        "expected_source": expected_source,
        "expected_route": expected_route.value if expected_route else "",
        "predicted_route": predicted_route,
        "route_source": route_source,
        "route_reasoning": route_reasoning,
        "correct": correct,
        "error_message": error_message,
    }


def summarize_planner_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize accuracy, confusion, and source breakdown per config."""
    config_names: list[str] = []
    for row in rows:
        name = str(row.get("config", ""))
        if name and name not in config_names:
            config_names.append(name)

    configs_summary: dict[str, Any] = {}
    for name in config_names:
        config_rows = [row for row in rows if row.get("config") == name]
        total = len(config_rows)
        correct = sum(1 for row in config_rows if bool(row.get("correct")))
        configs_summary[name] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "route_source_breakdown": _source_breakdown(config_rows),
            "per_route_accuracy": _per_route_accuracy(config_rows),
            "confusion": _confusion(config_rows),
        }

    return {"configs": configs_summary}


def _source_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for row in rows:
        source = str(row.get("route_source", "")) or "unknown"
        breakdown[source] = breakdown.get(source, 0) + 1
    return breakdown


def _per_route_accuracy(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for route_value in ALL_ROUTES:
        route_rows = [
            row for row in rows if row.get("expected_route") == route_value
        ]
        total = len(route_rows)
        if total == 0:
            continue
        correct = sum(1 for row in route_rows if bool(row.get("correct")))
        result[route_value] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
        }
    return result


def _confusion(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        expected = str(row.get("expected_route", "")) or "unknown"
        predicted = str(row.get("predicted_route", "")) or "unknown"
        matrix.setdefault(expected, {})
        matrix[expected][predicted] = matrix[expected].get(predicted, 0) + 1
    return matrix


def write_planner_results(
    rows: list[dict[str, Any]],
    output_path: str | Path = DEFAULT_CSV_OUTPUT_PATH,
) -> None:
    """Write per-question per-config planner rows to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PLANNER_EVAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: row.get(column, "") for column in PLANNER_EVAL_COLUMNS}
            )


def write_planner_summary(
    summary: dict[str, Any],
    output_path: str | Path = DEFAULT_SUMMARY_OUTPUT_PATH,
) -> None:
    """Write the planner summary JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_planner_questions(
    path: str | Path = DEFAULT_QUESTIONS_PATH,
) -> list[dict[str, Any]]:
    """Load the finance gold question set used for planner evaluation."""
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("Planner questions file must contain a JSON list")
    return questions
