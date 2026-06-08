"""LangGraph-backed Q&A analytics workflow."""

from __future__ import annotations

from contextlib import contextmanager
import warnings
from time import perf_counter
from typing import Any, TypedDict

from local_agentic_analytics.core.dataset_profile import (
    profile_to_compact_sql_context,
    profile_to_prompt_context,
)
from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.graph.workflow import (
    DEFAULT_TABLE_NAME,
    SequentialAnalyticsWorkflow,
    _collect_ollama_metrics,
    dataframe_to_compact_result,
)

warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change.*",
)


@contextmanager
def _suppress_langgraph_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The default value of `allowed_objects` will change.*",
        )
        yield


try:
    with _suppress_langgraph_warnings():
        from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - exercised only when dependency is absent.
    END = "__end__"
    START = "__start__"
    StateGraph = None


class _LangGraphWorkflowState(TypedDict, total=False):
    user_query: str
    domain: str
    analytics_state: AnalyticsState
    result_df: Any
    has_rule_based_sql: bool
    needs_repair: bool
    should_finalize: bool


class LangGraphAnalyticsWorkflow(SequentialAnalyticsWorkflow):
    """Run the Q&A pipeline through LangGraph with sequential node edges."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._compiled_graph = self._build_graph()

    def run(self, user_query: str) -> AnalyticsState:
        """Execute the LangGraph Q&A workflow and return the final state."""
        fallback_state = AnalyticsState(user_query=user_query)
        total_start = perf_counter()

        try:
            with _suppress_langgraph_warnings():
                graph_state = self._compiled_graph.invoke(
                    {
                        "user_query": user_query,
                        "domain": self.domain,
                    }
                )
            state = graph_state.get("analytics_state")
            if not isinstance(state, AnalyticsState):
                fallback_state.success = False
                fallback_state.error_message = (
                    "LangGraph workflow did not return AnalyticsState."
                )
                return fallback_state
            return state
        except Exception as exc:
            fallback_state.success = False
            fallback_state.error_message = str(exc)
            return fallback_state
        finally:
            candidate_state = locals().get("state")
            state_for_total = (
                candidate_state
                if isinstance(candidate_state, AnalyticsState)
                else fallback_state
            )
            state_for_total.latency["total"] = perf_counter() - total_start

    def _build_graph(self):
        if StateGraph is None:
            raise RuntimeError(
                "LangGraph is not installed. Install requirements.txt to use "
                "LangGraphAnalyticsWorkflow."
            )

        with _suppress_langgraph_warnings():
            graph = StateGraph(_LangGraphWorkflowState)
            graph.add_node("initialize_state", self.initialize_state_node)
            graph.add_node("load_profile", self.load_profile_node)
            graph.add_node("load_schema", self.load_schema_node)
            graph.add_node(
                "resolve_rule_based_sql",
                self.resolve_rule_based_sql_node,
            )
            graph.add_node("generate_sql", self.generate_sql_node)
            graph.add_node("semantic_validate_sql", self.semantic_validate_sql_node)
            graph.add_node("execute_sql", self.execute_sql_node)
            graph.add_node("repair_sql", self.repair_sql_node)
            graph.add_node("execute_repaired_sql", self.execute_repaired_sql_node)
            graph.add_node("generate_answer", self.generate_answer_node)
            graph.add_node("finalize", self.finalize_node)

            graph.add_edge(START, "initialize_state")
            graph.add_edge("initialize_state", "load_profile")
            graph.add_conditional_edges(
                "load_profile",
                self._route_after_load_profile,
                {
                    "load_schema": "load_schema",
                    "finalize": "finalize",
                },
            )
            graph.add_conditional_edges(
                "load_schema",
                self._route_after_load_schema,
                {
                    "resolve_rule_based_sql": "resolve_rule_based_sql",
                    "finalize": "finalize",
                },
            )
            graph.add_conditional_edges(
                "resolve_rule_based_sql",
                self._route_after_rule_based_sql,
                {
                    "generate_sql": "generate_sql",
                    "semantic_validate_sql": "semantic_validate_sql",
                    "finalize": "finalize",
                },
            )
            graph.add_conditional_edges(
                "generate_sql",
                self._route_after_generate_sql,
                {
                    "semantic_validate_sql": "semantic_validate_sql",
                    "finalize": "finalize",
                },
            )
            graph.add_conditional_edges(
                "semantic_validate_sql",
                self._route_after_semantic_validation,
                {
                    "execute_sql": "execute_sql",
                    "repair_sql": "repair_sql",
                    "finalize": "finalize",
                },
            )
            graph.add_conditional_edges(
                "execute_sql",
                self._route_after_sql_execution,
                {
                    "generate_answer": "generate_answer",
                    "repair_sql": "repair_sql",
                    "finalize": "finalize",
                },
            )
            graph.add_conditional_edges(
                "repair_sql",
                self._route_after_repair,
                {
                    "execute_repaired_sql": "execute_repaired_sql",
                    "finalize": "finalize",
                },
            )
            graph.add_conditional_edges(
                "execute_repaired_sql",
                self._route_after_repaired_execution,
                {
                    "generate_answer": "generate_answer",
                    "finalize": "finalize",
                },
            )
            graph.add_edge("generate_answer", "finalize")
            graph.add_edge("finalize", END)

            return graph.compile()

    def initialize_state_node(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> _LangGraphWorkflowState:
        start = perf_counter()
        user_query = graph_state.get("user_query", "")
        state = AnalyticsState(user_query=user_query)
        state.latency["initialize_state"] = perf_counter() - start
        return {
            **graph_state,
            "analytics_state": state,
            "should_finalize": False,
            "needs_repair": False,
        }

    def load_profile_node(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> _LangGraphWorkflowState:
        state = _get_analytics_state(graph_state)
        start = perf_counter()
        try:
            self.full_dataset_profile_context = profile_to_prompt_context(
                self.dataset_profile
            )
            self.dataset_profile_context = profile_to_compact_sql_context(
                self.dataset_profile
            )
            if not self.table_name:
                self.table_name = self.dataset_profile.table_name or DEFAULT_TABLE_NAME
            return {**graph_state, "analytics_state": state}
        except Exception as exc:
            return self._mark_finalize_error(graph_state, exc)
        finally:
            state.latency["load_profile"] = perf_counter() - start

    def load_schema_node(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> _LangGraphWorkflowState:
        state = _get_analytics_state(graph_state)
        try:
            state.schema = self._run_step(
                state,
                "load_schema",
                "duckdb",
                "schema",
                "duckdb.schema",
                lambda: self.duckdb_tool.get_schema(self.table_name),
                input_summary=f"table_name={self.table_name}",
            )
            return {**graph_state, "analytics_state": state}
        except Exception as exc:
            return self._mark_finalize_error(graph_state, exc)

    def resolve_rule_based_sql_node(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> _LangGraphWorkflowState:
        state = _get_analytics_state(graph_state)
        try:
            resolved_sql = self._run_step(
                state,
                "resolve_rule_based_sql",
                "rule_based_sql_resolver",
                "resolve",
                "rule_based_sql_resolver.resolve",
                lambda: self.rule_based_sql_resolver.resolve(
                    question=state.user_query,
                    dataset_profile=self.dataset_profile,
                ),
                input_summary=f"question={state.user_query}",
                status_from_result=lambda result: "success" if result else "no_match",
            )
            if resolved_sql:
                state.generated_sql = resolved_sql
                state.route = "rule_based_sql"
            else:
                state.route = "llm_sql"
            return {
                **graph_state,
                "analytics_state": state,
                "has_rule_based_sql": bool(resolved_sql),
            }
        except Exception as exc:
            return self._mark_finalize_error(graph_state, exc)

    def generate_sql_node(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> _LangGraphWorkflowState:
        state = _get_analytics_state(graph_state)
        try:
            state.generated_sql = self._run_step(
                state,
                "generate_sql",
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
            return {**graph_state, "analytics_state": state}
        except Exception as exc:
            return self._mark_finalize_error(graph_state, exc)

    def semantic_validate_sql_node(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> _LangGraphWorkflowState:
        state = _get_analytics_state(graph_state)
        try:
            self._run_step(
                state,
                "semantic_validate_sql",
                "sql_semantic_guard",
                "validate",
                "sql_semantic_guard.validate",
                lambda: self._validate_sql_semantics(
                    state.user_query,
                    state.generated_sql or "",
                ),
                input_summary=state.generated_sql or "",
            )
            return {
                **graph_state,
                "analytics_state": state,
                "needs_repair": False,
            }
        except Exception as exc:
            state.error_message = str(exc)
            state.route = f"{state.route or 'unknown'}_with_repair"
            return {
                **graph_state,
                "analytics_state": state,
                "needs_repair": True,
            }

    def execute_sql_node(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> _LangGraphWorkflowState:
        state = _get_analytics_state(graph_state)
        try:
            result_df = self._run_step(
                state,
                "execute_sql",
                "duckdb",
                "query",
                "duckdb.query",
                lambda: self.duckdb_tool.execute_query(state.generated_sql or ""),
                input_summary=state.generated_sql or "",
            )
            return {
                **graph_state,
                "analytics_state": state,
                "result_df": result_df,
                "needs_repair": False,
            }
        except Exception as exc:
            state.error_message = str(exc)
            if not (state.route or "").endswith("_with_repair"):
                state.route = f"{state.route or 'unknown'}_with_repair"
            return {
                **graph_state,
                "analytics_state": state,
                "needs_repair": True,
            }

    def repair_sql_node(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> _LangGraphWorkflowState:
        state = _get_analytics_state(graph_state)
        try:
            state.repaired_sql = self._run_step(
                state,
                "repair_sql",
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
            return {
                **graph_state,
                "analytics_state": state,
                "needs_repair": False,
            }
        except Exception as exc:
            return self._mark_finalize_error(graph_state, exc)

    def execute_repaired_sql_node(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> _LangGraphWorkflowState:
        state = _get_analytics_state(graph_state)
        try:
            self._run_step(
                state,
                "semantic_validate_repaired_sql",
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
                "execute_repaired_sql",
                "duckdb",
                "query_repaired",
                "duckdb.query_repaired",
                lambda: self.duckdb_tool.execute_query(state.repaired_sql or ""),
                input_summary=state.repaired_sql or "",
            )
            return {
                **graph_state,
                "analytics_state": state,
                "result_df": result_df,
                "needs_repair": False,
            }
        except Exception as exc:
            return self._mark_finalize_error(graph_state, exc)

    def generate_answer_node(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> _LangGraphWorkflowState:
        state = _get_analytics_state(graph_state)
        result_df = graph_state.get("result_df")
        if result_df is None:
            return self._mark_finalize_error(
                graph_state,
                RuntimeError("LangGraph workflow did not produce a SQL result."),
            )

        try:
            state.sql_result = dataframe_to_compact_result(
                result_df,
                max_rows=self.max_report_rows,
            )
            executed_sql = state.repaired_sql or state.generated_sql or ""
            state.final_answer = self._run_step(
                state,
                "generate_answer",
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
            return {**graph_state, "analytics_state": state}
        except Exception as exc:
            return self._mark_finalize_error(graph_state, exc)

    def finalize_node(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> _LangGraphWorkflowState:
        state = _get_analytics_state(graph_state)
        start = perf_counter()
        state.success = bool(state.final_answer and state.sql_result is not None)
        state.latency["finalize"] = perf_counter() - start
        return {**graph_state, "analytics_state": state}

    def _route_after_rule_based_sql(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> str:
        if graph_state.get("should_finalize"):
            return "finalize"
        if graph_state.get("has_rule_based_sql"):
            return "semantic_validate_sql"
        return "generate_sql"

    def _route_after_load_profile(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> str:
        if graph_state.get("should_finalize"):
            return "finalize"
        return "load_schema"

    def _route_after_load_schema(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> str:
        if graph_state.get("should_finalize"):
            return "finalize"
        return "resolve_rule_based_sql"

    def _route_after_generate_sql(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> str:
        if graph_state.get("should_finalize"):
            return "finalize"
        return "semantic_validate_sql"

    def _route_after_semantic_validation(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> str:
        if graph_state.get("should_finalize"):
            return "finalize"
        if graph_state.get("needs_repair"):
            return "repair_sql"
        return "execute_sql"

    def _route_after_sql_execution(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> str:
        if graph_state.get("should_finalize"):
            return "finalize"
        if graph_state.get("needs_repair"):
            return "repair_sql"
        return "generate_answer"

    def _route_after_repair(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> str:
        if graph_state.get("should_finalize"):
            return "finalize"
        return "execute_repaired_sql"

    def _route_after_repaired_execution(
        self,
        graph_state: _LangGraphWorkflowState,
    ) -> str:
        if graph_state.get("should_finalize"):
            return "finalize"
        return "generate_answer"

    def _mark_finalize_error(
        self,
        graph_state: _LangGraphWorkflowState,
        exc: Exception,
    ) -> _LangGraphWorkflowState:
        state = _get_analytics_state(graph_state)
        state.success = False
        state.error_message = str(exc)
        return {
            **graph_state,
            "analytics_state": state,
            "needs_repair": False,
            "should_finalize": True,
        }


def _get_analytics_state(graph_state: _LangGraphWorkflowState) -> AnalyticsState:
    state = graph_state.get("analytics_state")
    if not isinstance(state, AnalyticsState):
        raise RuntimeError("LangGraph workflow state is missing AnalyticsState.")
    return state


def run_langgraph_workflow(
    user_query: str,
    domain: str = "energy",
) -> AnalyticsState:
    """Run the default LangGraph Q&A workflow for one user query."""
    workflow = LangGraphAnalyticsWorkflow(domain=domain)
    return workflow.run(user_query)
