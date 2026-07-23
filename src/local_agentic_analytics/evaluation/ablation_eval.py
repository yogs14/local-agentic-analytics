"""Ablation evaluation: isolate model vs deterministic scaffolding accuracy.

This runs the energy gold SQL set under a series of named configurations that
progressively enable the deterministic scaffolding (rule-based resolver, domain
SQL normalizers, semantic guard, repair agent). It is a measurement tool: it
must not tune or alter any of the components it toggles.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from local_agentic_analytics.core.config import PROJECT_ROOT
from local_agentic_analytics.core.pipeline_toggles import PipelineToggles
from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.evaluation.result_comparison import compare_result_sets
from local_agentic_analytics.evaluation.sql_gold_eval import (
    DEFAULT_GOLD_QUESTIONS_PATH,
    SqlExecutionResult,
    compare_numeric_results,
    execute_sql,
    load_gold_questions,
    load_gold_sql,
    _load_default_duckdb_tool,
)
from local_agentic_analytics.graph.workflow import SequentialAnalyticsWorkflow
from local_agentic_analytics.tools.duckdb_tool import DuckDBTool


DEFAULT_CSV_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "ablation_eval.csv"
)
DEFAULT_SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "ablation_summary.json"
)

# "numeric_match" is the legacy scalar-only metric; "result_match_full" is the
# Fase 2 row-set comparison. Both are reported side by side.
ABLATION_EVAL_COLUMNS = (
    "config",
    "question_id",
    "question",
    "route",
    "agent_sql",
    "raw_generated_sql",
    "execution_success",
    "gold_success",
    "numeric_match",
    "absolute_error",
    "relative_error",
    "error_message",
    "result_match_full",
    "result_match_reason",
)

FULL_CONFIG_NAME = "D_full"


@dataclass(frozen=True)
class AblationConfig:
    """One named pipeline configuration to evaluate."""

    name: str
    toggles: PipelineToggles


ABLATION_CONFIGS: tuple[AblationConfig, ...] = (
    AblationConfig(
        name="A_raw_llm",
        toggles=PipelineToggles(
            use_rule_based_resolver=False,
            apply_domain_normalization=False,
            use_semantic_guard=False,
            use_repair=False,
        ),
    ),
    AblationConfig(
        name="B_llm_norm",
        toggles=PipelineToggles(
            use_rule_based_resolver=False,
            apply_domain_normalization=True,
            use_semantic_guard=False,
            use_repair=False,
        ),
    ),
    AblationConfig(
        name="C_llm_norm_guard_repair",
        toggles=PipelineToggles(
            use_rule_based_resolver=False,
            apply_domain_normalization=True,
            use_semantic_guard=True,
            use_repair=True,
        ),
    ),
    AblationConfig(
        name=FULL_CONFIG_NAME,
        toggles=PipelineToggles(
            use_rule_based_resolver=True,
            apply_domain_normalization=True,
            use_semantic_guard=True,
            use_repair=True,
        ),
    ),
)


WorkflowFactory = Callable[[PipelineToggles], Any]


def _default_workflow_factory(toggles: PipelineToggles) -> SequentialAnalyticsWorkflow:
    return SequentialAnalyticsWorkflow(toggles=toggles)


def run_ablation_evaluation(
    questions: list[dict[str, Any]],
    workflow_factory: WorkflowFactory | None = None,
    duckdb_tool: DuckDBTool | None = None,
    configs: tuple[AblationConfig, ...] = ABLATION_CONFIGS,
) -> list[dict[str, Any]]:
    """Run every config against every question and return flat result rows."""
    workflow_factory = workflow_factory or _default_workflow_factory
    duckdb_tool = duckdb_tool or _load_default_duckdb_tool()

    rows: list[dict[str, Any]] = []
    for config in configs:
        workflow = workflow_factory(config.toggles)
        for question in questions:
            try:
                rows.append(
                    run_single_ablation_question(
                        question=question,
                        config=config,
                        workflow=workflow,
                        duckdb_tool=duckdb_tool,
                    )
                )
            except Exception as exc:
                rows.append(_failed_row(config.name, question, str(exc)))

    return rows


def run_single_ablation_question(
    question: dict[str, Any],
    config: AblationConfig,
    workflow: Any,
    duckdb_tool: DuckDBTool,
) -> dict[str, Any]:
    """Run one gold question under one config and capture comparison fields."""
    question_id = str(question["id"])
    question_text = str(question["question"])
    errors: list[str] = []

    try:
        gold_sql = load_gold_sql(question["gold_sql_file"])
    except Exception as exc:
        gold_sql = ""
        errors.append(f"gold_sql_load: {exc}")

    try:
        state = workflow.run(question_text)
    except Exception as exc:
        state = AnalyticsState(
            user_query=question_text,
            error_message=str(exc),
            success=False,
        )

    if state.error_message and not state.success:
        errors.append(f"workflow: {state.error_message}")

    agent_sql = state.repaired_sql or state.generated_sql or ""
    if agent_sql:
        agent_result = execute_sql(duckdb_tool, agent_sql)
    else:
        agent_result = SqlExecutionResult(
            success=False,
            result_text="",
            error_message="Agent did not produce SQL",
        )

    if gold_sql:
        gold_result = execute_sql(duckdb_tool, gold_sql)
    else:
        gold_result = SqlExecutionResult(
            success=False,
            result_text="",
            error_message="Gold SQL was not loaded",
        )

    if agent_result.error_message:
        errors.append(f"agent_sql: {agent_result.error_message}")
    if gold_result.error_message:
        errors.append(f"gold_sql: {gold_result.error_message}")

    comparison = compare_numeric_results(
        agent_result.numeric_value,
        gold_result.numeric_value,
    )
    result_comparison = compare_result_sets(
        agent_result.result_text,
        gold_result.result_text,
    )

    return {
        "config": config.name,
        "question_id": question_id,
        "question": question_text,
        "route": state.route or "",
        "agent_sql": agent_sql,
        "raw_generated_sql": state.raw_generated_sql or "",
        "execution_success": agent_result.success,
        "gold_success": gold_result.success,
        "numeric_match": comparison["numeric_match"],
        "absolute_error": comparison["absolute_error"],
        "relative_error": comparison["relative_error"],
        "error_message": " | ".join(errors),
        "result_match_full": result_comparison["result_match_full"],
        "result_match_reason": result_comparison["result_match_reason"],
    }


def _failed_row(
    config_name: str,
    question: dict[str, Any],
    error_message: str,
) -> dict[str, Any]:
    return {
        "config": config_name,
        "question_id": str(question.get("id", "")),
        "question": str(question.get("question", "")),
        "route": "",
        "agent_sql": "",
        "raw_generated_sql": "",
        "execution_success": False,
        "gold_success": False,
        "numeric_match": "",
        "absolute_error": "",
        "relative_error": "",
        "error_message": error_message,
        "result_match_full": "",
        "result_match_reason": "not_evaluated",
    }


def classify_route(route: str) -> str:
    """Classify a state route into rule_based / llm / other."""
    if route.startswith("rule_based_sql"):
        return "rule_based"
    if route.startswith("llm_sql"):
        return "llm"
    return "other"


def summarize_ablation_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize ablation rows per config plus the D_full route distribution."""
    config_names: list[str] = []
    for row in rows:
        name = str(row.get("config", ""))
        if name and name not in config_names:
            config_names.append(name)

    configs_summary: dict[str, dict[str, float | int]] = {}
    for name in config_names:
        config_rows = [row for row in rows if row.get("config") == name]
        total = len(config_rows)
        execution_success_count = sum(
            1 for row in config_rows if bool(row.get("execution_success"))
        )
        numeric_rows = [
            row for row in config_rows if row.get("numeric_match") != ""
        ]
        numeric_match_count = sum(
            1 for row in numeric_rows if bool(row.get("numeric_match"))
        )
        result_full_rows = [
            row for row in config_rows if row.get("result_match_full") != ""
        ]
        result_full_match_count = sum(
            1 for row in result_full_rows if bool(row.get("result_match_full"))
        )
        configs_summary[name] = {
            "total": total,
            "execution_success_count": execution_success_count,
            "execution_success_rate": (
                execution_success_count / total if total else 0.0
            ),
            "numeric_compared_count": len(numeric_rows),
            "numeric_match_count": numeric_match_count,
            "numeric_match_rate": numeric_match_count / total if total else 0.0,
            "result_full_compared_count": len(result_full_rows),
            "result_full_match_count": result_full_match_count,
            "result_match_full_rate": (
                result_full_match_count / total if total else 0.0
            ),
        }

    return {
        "configs": configs_summary,
        "d_full_route_distribution": _route_distribution(rows, FULL_CONFIG_NAME),
    }


def _route_distribution(
    rows: list[dict[str, Any]],
    config_name: str,
) -> dict[str, float | int]:
    config_rows = [row for row in rows if row.get("config") == config_name]
    total = len(config_rows)
    rule_based_count = sum(
        1 for row in config_rows if classify_route(str(row.get("route", ""))) == "rule_based"
    )
    llm_count = sum(
        1 for row in config_rows if classify_route(str(row.get("route", ""))) == "llm"
    )
    other_count = total - rule_based_count - llm_count
    return {
        "total": total,
        "rule_based_count": rule_based_count,
        "llm_count": llm_count,
        "other_count": other_count,
        "rule_based_pct": rule_based_count / total if total else 0.0,
        "llm_pct": llm_count / total if total else 0.0,
    }


def write_ablation_results(
    rows: list[dict[str, Any]],
    output_path: str | Path = DEFAULT_CSV_OUTPUT_PATH,
) -> None:
    """Write per-question per-config ablation rows to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ABLATION_EVAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: row.get(column, "") for column in ABLATION_EVAL_COLUMNS}
            )


def write_ablation_summary(
    summary: dict[str, Any],
    output_path: str | Path = DEFAULT_SUMMARY_OUTPUT_PATH,
) -> None:
    """Write the ablation summary JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_default_gold_questions(
    path: str | Path = DEFAULT_GOLD_QUESTIONS_PATH,
) -> list[dict[str, Any]]:
    """Load the default energy gold question set."""
    return load_gold_questions(path)
