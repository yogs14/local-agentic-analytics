"""Sequential text-to-SQL analytics workflow."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TypeVar

import pandas as pd

from local_agentic_analytics.agents.repair_agent import SQLRepairAgent
from local_agentic_analytics.agents.reporter_agent import ReporterAgent
from local_agentic_analytics.agents.rule_based_sql_resolver import (
    RuleBasedSQLResolver,
)
from local_agentic_analytics.agents.sql_agent import SQLAgent
from local_agentic_analytics.core.config import PROJECT_ROOT, load_config
from local_agentic_analytics.core.pipeline_toggles import PipelineToggles
from local_agentic_analytics.core.dataset_profile import (
    DatasetProfile,
    load_dataset_profile,
    profile_to_compact_sql_context,
    profile_to_prompt_context,
)
from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.domain.adapters import DomainAdapter
from local_agentic_analytics.domain.registry import get_domain_adapter
from local_agentic_analytics.evaluation.audit_logger import (
    ToolAuditLogger,
    summarize_value,
)
from local_agentic_analytics.tools.duckdb_tool import DuckDBTool
from local_agentic_analytics.tools.sql_semantic_guard import (
    validate_energy_sql_semantics,
)


T = TypeVar("T")
DEFAULT_TABLE_NAME = "electric_power"
DEFAULT_MAX_REPORT_ROWS = 20


class SequentialAnalyticsWorkflow:
    """Run the local analytics pipeline step by step on one thread."""

    def __init__(
        self,
        duckdb_tool: DuckDBTool | None = None,
        sql_agent: SQLAgent | None = None,
        rule_based_sql_resolver: RuleBasedSQLResolver | None = None,
        repair_agent: SQLRepairAgent | None = None,
        reporter_agent: ReporterAgent | None = None,
        domain: str = "energy",
        dataset_profile: DatasetProfile | None = None,
        table_name: str | None = None,
        max_report_rows: int = DEFAULT_MAX_REPORT_ROWS,
        domain_adapter: DomainAdapter | None = None,
        audit_logger: ToolAuditLogger | None = None,
        toggles: PipelineToggles | None = None,
    ):
        if table_name is not None and not table_name.strip():
            raise ValueError("table_name must not be empty")
        if max_report_rows < 1:
            raise ValueError("max_report_rows must be greater than 0")

        self.toggles = toggles or PipelineToggles()
        self.domain = domain.strip() if domain else "energy"
        # Energy returns None here (prompt path unchanged); finance auto-loads
        # its adapter so the domain few-shots reach the SQL agent.
        self.domain_adapter = (
            domain_adapter
            if domain_adapter is not None
            else get_domain_adapter(self.domain)
        )
        self.dataset_profile = dataset_profile or load_dataset_profile(self.domain)
        self.full_dataset_profile_context = profile_to_prompt_context(
            self.dataset_profile
        )
        self.dataset_profile_context = profile_to_compact_sql_context(
            self.dataset_profile
        )
        resolved_table_name = table_name or self.dataset_profile.table_name
        if not resolved_table_name or not resolved_table_name.strip():
            resolved_table_name = DEFAULT_TABLE_NAME

        self.duckdb_tool = duckdb_tool or DuckDBTool(str(_load_default_db_path()))
        self.sql_agent = sql_agent or SQLAgent(
            dataset_profile_context=self.dataset_profile_context,
            domain_adapter=self.domain_adapter,
            apply_domain_normalization=self.toggles.apply_domain_normalization,
        )
        self.rule_based_sql_resolver = (
            rule_based_sql_resolver or RuleBasedSQLResolver()
        )
        self.repair_agent = repair_agent or SQLRepairAgent(
            apply_domain_normalization=self.toggles.apply_domain_normalization,
        )
        self.reporter_agent = reporter_agent or ReporterAgent()
        self.table_name = resolved_table_name.strip()
        self.max_report_rows = max_report_rows
        self.audit_logger = audit_logger or ToolAuditLogger()

    def run(self, user_query: str) -> AnalyticsState:
        """Execute the full sequential workflow and return the final state."""
        state = AnalyticsState(user_query=user_query)
        total_start = perf_counter()

        try:
            state.schema = self._run_step(
                state,
                "schema",
                "duckdb",
                "schema",
                "duckdb.schema",
                lambda: self.duckdb_tool.get_schema(self.table_name),
                input_summary=f"table_name={self.table_name}",
            )
            if self.toggles.use_rule_based_resolver:
                resolved_sql = self._run_step(
                    state,
                    "rule_based_sql_resolution",
                    "rule_based_sql_resolver",
                    "resolve",
                    "rule_based_sql_resolver.resolve",
                    lambda: self.rule_based_sql_resolver.resolve(
                        question=state.user_query,
                        dataset_profile=self.dataset_profile,
                    ),
                    input_summary=f"question={state.user_query}",
                    status_from_result=lambda result: (
                        "success" if result else "no_match"
                    ),
                )
            else:
                resolved_sql = None

            if resolved_sql:
                state.generated_sql = resolved_sql
                state.route = "rule_based_sql"
            else:
                state.route = "llm_sql"
                state.generated_sql = self._run_step(
                    state,
                    "sql_generation",
                    "ollama",
                    "sql_generation",
                    "ollama.sql_generation",
                    lambda: self.sql_agent.generate_sql(
                        question=state.user_query,
                        schema=state.schema or "",
                        dataset_profile_context=self.dataset_profile_context,
                    ),
                    input_summary=f"question={state.user_query}",
                    ollama_action="sql_generation",
                    ollama_metrics_provider=lambda: _collect_ollama_metrics(
                        self.sql_agent
                    ),
                )
                state.raw_generated_sql = getattr(
                    self.sql_agent, "last_raw_generated_sql", None
                )

            try:
                if self.toggles.use_semantic_guard:
                    self._run_step(
                        state,
                        "sql_semantic_validation",
                        "sql_semantic_guard",
                        "validate",
                        "sql_semantic_guard.validate",
                        lambda: self._validate_sql_semantics(
                            state.user_query,
                            state.generated_sql or "",
                        ),
                        input_summary=state.generated_sql or "",
                    )
                result_df = self._run_step(
                    state,
                    "sql_execution",
                    "duckdb",
                    "query",
                    "duckdb.query",
                    lambda: self.duckdb_tool.execute_query(state.generated_sql or ""),
                    input_summary=state.generated_sql or "",
                )
            except Exception as exc:
                state.error_message = str(exc)
                if not self.toggles.use_repair:
                    raise
                state.route = f"{state.route or 'unknown'}_with_repair"
                state.repaired_sql = self._run_step(
                    state,
                    "sql_repair",
                    "ollama",
                    "sql_repair",
                    "ollama.sql_repair",
                    lambda: self.repair_agent.repair_sql(
                        failed_sql=state.generated_sql or "",
                        error_message=state.error_message or "",
                        schema=state.schema or "",
                        repair_attempted=False,
                        user_question=state.user_query,
                    ),
                    input_summary=state.error_message or "",
                    ollama_action="sql_repair",
                    ollama_metrics_provider=lambda: _collect_ollama_metrics(
                        self.repair_agent
                    ),
                )
                if self.toggles.use_semantic_guard:
                    self._run_step(
                        state,
                        "repair_semantic_validation",
                        "sql_semantic_guard",
                        "validate_repaired",
                        "sql_semantic_guard.validate_repaired",
                        lambda: self._validate_sql_semantics(
                            state.user_query,
                            state.repaired_sql or "",
                        ),
                        input_summary=state.repaired_sql or "",
                    )
                result_df = self._run_step(
                    state,
                    "repair_execution",
                    "duckdb",
                    "query_repaired",
                    "duckdb.query_repaired",
                    lambda: self.duckdb_tool.execute_query(state.repaired_sql or ""),
                    input_summary=state.repaired_sql or "",
                )

            state.sql_result = dataframe_to_compact_result(
                result_df,
                max_rows=self.max_report_rows,
            )
            executed_sql = state.repaired_sql or state.generated_sql or ""
            state.final_answer = self._run_step(
                state,
                "reporting",
                "ollama",
                "reporting",
                "ollama.reporting",
                lambda: self.reporter_agent.generate_answer(
                    question=state.user_query,
                    sql=executed_sql,
                    query_result=state.sql_result,
                ),
                input_summary=executed_sql,
                ollama_action="reporting",
                ollama_metrics_provider=lambda: _collect_ollama_metrics(
                    self.reporter_agent
                ),
            )
            state.success = True
            return state
        except Exception as exc:
            state.success = False
            state.error_message = str(exc)
            return state
        finally:
            state.latency["total"] = perf_counter() - total_start

    def _validate_sql_semantics(self, question: str, sql: str) -> str:
        if self.dataset_profile.domain.lower() != "energy":
            return "skipped"

        is_valid, message = validate_energy_sql_semantics(question, sql)
        if not is_valid:
            raise ValueError(f"Semantic SQL validation failed: {message}")
        return "valid"

    def _run_step(
        self,
        state: AnalyticsState,
        name: str,
        component: str,
        action: str,
        tool: str,
        func: Callable[[], T],
        input_summary: str = "",
        ollama_action: str | None = None,
        ollama_metrics_provider: Callable[[], dict[str, Any]] | None = None,
        status_from_result: Callable[[T | None], str] | None = None,
    ) -> T:
        start = perf_counter()
        status = "success"
        error_message = ""
        result: T | None = None
        try:
            result = func()
            return result
        except Exception as exc:
            status = "error"
            error_message = str(exc)
            raise
        finally:
            elapsed = perf_counter() - start
            state.latency[name] = elapsed
            metadata = {}
            if ollama_action and ollama_metrics_provider is not None:
                metadata = _safe_collect_metrics(ollama_metrics_provider)
            if status == "success" and status_from_result is not None:
                status = status_from_result(result)

            output_summary = summarize_value(result)
            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            event = self.audit_logger.log(
                component=component,
                action=action,
                tool=tool,
                status=status,
                latency_seconds=elapsed,
                input_summary=input_summary,
                output_summary=output_summary,
                error_message=error_message,
                metadata=metadata,
                timestamp=timestamp,
            )
            if not isinstance(event, dict):
                event = _build_tool_call_record(
                    timestamp=timestamp,
                    component=component,
                    action=action,
                    tool=tool,
                    status=status,
                    latency_seconds=elapsed,
                    input_summary=input_summary,
                    output_summary=output_summary,
                    error_message=error_message,
                    metadata=metadata,
                )
            state.tool_calls.append(event)
            _add_selected_tool(state, tool)


def _collect_ollama_metrics(agent: Any) -> dict[str, Any]:
    ollama_tool = getattr(agent, "ollama_tool", None)
    get_last_metrics = getattr(ollama_tool, "get_last_metrics", None)
    if not callable(get_last_metrics):
        return {}

    metrics = get_last_metrics()
    if not isinstance(metrics, dict):
        return {}
    return metrics


def _safe_collect_metrics(
    metrics_provider: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        metrics = metrics_provider()
    except Exception:
        return {}
    if not isinstance(metrics, dict):
        return {}
    return metrics


def _build_tool_call_record(
    timestamp: str,
    component: str,
    action: str,
    tool: str,
    status: str,
    latency_seconds: float,
    input_summary: str,
    output_summary: str,
    error_message: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "component": component,
        "action": action,
        "tool": tool,
        "status": status,
        "latency_seconds": round(float(latency_seconds), 6),
        "input_summary": input_summary,
        "output_summary": output_summary,
        "error_message": error_message,
        "metadata": metadata,
    }


def _add_selected_tool(state: AnalyticsState, tool: str) -> None:
    if tool not in state.selected_tools:
        state.selected_tools.append(tool)


def dataframe_to_compact_result(
    dataframe: pd.DataFrame,
    max_rows: int = DEFAULT_MAX_REPORT_ROWS,
) -> dict[str, Any]:
    """Convert a query DataFrame into a compact JSON-serializable result."""
    if max_rows < 1:
        raise ValueError("max_rows must be greater than 0")

    preview = dataframe.head(max_rows)
    records = json.loads(preview.to_json(orient="records", date_format="iso"))

    return {
        "row_count": int(len(dataframe)),
        "columns": [str(column) for column in dataframe.columns],
        "rows": records,
        "truncated": len(dataframe) > max_rows,
    }


def _load_default_db_path() -> Path:
    config = load_config("duckdb.yaml")
    db_path = config.get("duckdb", {}).get("database_path")

    if not isinstance(db_path, str) or not db_path.strip():
        return PROJECT_ROOT / "databases" / "duckdb" / "analytics.duckdb"

    path = Path(db_path)
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def run_workflow(user_query: str) -> AnalyticsState:
    """Convenience function for running the default sequential workflow."""
    workflow = SequentialAnalyticsWorkflow()
    return workflow.run(user_query)
