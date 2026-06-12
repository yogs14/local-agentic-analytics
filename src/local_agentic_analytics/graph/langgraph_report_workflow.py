"""LangGraph-backed energy report workflow."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, TypedDict
import warnings

from local_agentic_analytics.agents.insight_agent import InsightAgent
from local_agentic_analytics.evaluation.audit_logger import (
    DEFAULT_AUDIT_LOG_PATH,
    ToolAuditLogger,
    summarize_value,
)
from local_agentic_analytics.graph.report_workflow import (
    DEFAULT_DB_PATH,
    DEFAULT_FIGURES_DIR,
    DEFAULT_LOG_PATH,
    DEFAULT_PDF_DIR,
    DEFAULT_TEMPLATE_PATH,
    DEFAULT_TEX_PATH,
    EnergyReportWorkflow,
)
from local_agentic_analytics.reporting.latex_builder import render_latex_report
from local_agentic_analytics.reporting.pdf_compiler import compile_pdf
from local_agentic_analytics.reporting.report_schema import AnalysisReport
from local_agentic_analytics.visualization.chart_registry import (
    generate_all_energy_charts,
)


INSIGHT_FALLBACK = (
    "Insight otomatis tidak dapat dibuat untuk grafik ini. "
    "Periksa koneksi Ollama atau konfigurasi model lokal."
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


class ReportGraphState(TypedDict, total=False):
    domain: str
    timestamp_start: str
    timestamp_end: str
    charts: list[dict[str, Any]]
    chart_contexts: list[dict[str, Any]]
    insights: list[dict[str, Any]]
    report: AnalysisReport
    tex_path: Path
    pdf_path: Path
    log_path: Path
    error_message: str
    pdf_error: str
    latency: dict[str, float]
    tool_calls: list[dict[str, Any]]
    metadata: dict[str, Any]


class LangGraphReportWorkflow:
    """Generate an energy report through a sequential LangGraph graph."""

    def __init__(
        self,
        domain: str = "energy",
        db_path: str | Path = DEFAULT_DB_PATH,
        figures_dir: str | Path = DEFAULT_FIGURES_DIR,
        latex_output_path: str | Path = DEFAULT_TEX_PATH,
        pdf_dir: str | Path = DEFAULT_PDF_DIR,
        template_path: str | Path = DEFAULT_TEMPLATE_PATH,
        log_path: str | Path = DEFAULT_LOG_PATH,
        audit_logger: ToolAuditLogger | None = None,
        audit_log_path: str | Path = DEFAULT_AUDIT_LOG_PATH,
        insight_agent: InsightAgent | None = None,
    ):
        if domain != "energy":
            raise ValueError("LangGraphReportWorkflow currently supports only energy.")

        self.domain = domain
        self._workflow = EnergyReportWorkflow(
            db_path=db_path,
            figures_dir=figures_dir,
            latex_output_path=latex_output_path,
            pdf_dir=pdf_dir,
            template_path=template_path,
            log_path=log_path,
            insight_agent=insight_agent,
        )
        self.audit_logger = audit_logger or ToolAuditLogger(log_path=audit_log_path)
        self._compiled_graph = self._build_graph()

    def run(self) -> dict[str, Any]:
        """Run the report graph and return report metadata."""
        try:
            graph_state = self._compiled_graph.invoke({"domain": self.domain})
        except Exception as exc:
            graph_state = {
                "domain": self.domain,
                "timestamp_start": _utc_now(),
                "timestamp_end": _utc_now(),
                "error_message": str(exc),
                "pdf_error": "",
                "latency": {},
                "tool_calls": [],
                "charts": [],
                "insights": [],
            }

        metadata = graph_state.get("metadata")
        if isinstance(metadata, dict):
            return metadata

        return self._build_metadata(graph_state)

    def _build_graph(self):
        if StateGraph is None:
            raise RuntimeError(
                "LangGraph is not installed. Install requirements.txt to use "
                "LangGraphReportWorkflow."
            )

        with _suppress_langgraph_warnings():
            graph = StateGraph(ReportGraphState)
            graph.add_node("initialize_report_state", self.initialize_report_state)
            graph.add_node("generate_charts", self.generate_charts)
            graph.add_node("build_chart_contexts", self.build_chart_contexts)
            graph.add_node("generate_insights", self.generate_insights)
            graph.add_node("build_report", self.build_report)
            graph.add_node("render_latex", self.render_latex)
            graph.add_node("compile_pdf", self.compile_pdf)
            graph.add_node("write_report_log", self.write_report_log)
            graph.add_node("finalize_report", self.finalize_report)

            graph.add_edge(START, "initialize_report_state")
            graph.add_edge("initialize_report_state", "generate_charts")
            graph.add_edge("generate_charts", "build_chart_contexts")
            graph.add_edge("build_chart_contexts", "generate_insights")
            graph.add_edge("generate_insights", "build_report")
            graph.add_edge("build_report", "render_latex")
            graph.add_edge("render_latex", "compile_pdf")
            graph.add_edge("compile_pdf", "write_report_log")
            graph.add_edge("write_report_log", "finalize_report")
            graph.add_edge("finalize_report", END)

            return graph.compile()

    def initialize_report_state(self, state: ReportGraphState) -> ReportGraphState:
        start = perf_counter()
        next_state: ReportGraphState = {
            **state,
            "timestamp_start": _utc_now(),
            "charts": [],
            "chart_contexts": [],
            "insights": [],
            "error_message": "",
            "pdf_error": "",
            "latency": {},
            "tool_calls": [],
        }
        _record_latency(next_state, "initialize_report_state", start)
        return next_state

    def generate_charts(self, state: ReportGraphState) -> ReportGraphState:
        return self._run_blocking_node(
            state=state,
            node_name="generate_charts",
            component="report",
            action="generate_charts",
            tool="report.generate_charts",
            func=lambda: generate_all_energy_charts(
                db_path=self._workflow.db_path,
                output_dir=self._workflow.figures_dir,
            ),
            output_key="charts",
            input_summary=f"db_path={self._workflow.db_path}",
        )

    def build_chart_contexts(self, state: ReportGraphState) -> ReportGraphState:
        return self._run_blocking_node(
            state=state,
            node_name="build_chart_contexts",
            component="report",
            action="build_chart_contexts",
            tool="report.build_chart_contexts",
            func=lambda: self._workflow._build_chart_contexts(
                list(state.get("charts", []))
            ),
            output_key="chart_contexts",
            input_summary=f"chart_count={len(state.get('charts', []))}",
        )

    def generate_insights(self, state: ReportGraphState) -> ReportGraphState:
        node_start = perf_counter()
        if _has_blocking_error(state):
            elapsed = _record_latency(state, "generate_insights", node_start)
            _append_tool_call(
                state=state,
                audit_logger=self.audit_logger,
                component="ollama",
                action="generate_insights",
                tool="ollama.generate_insights",
                status="skipped",
                latency_seconds=elapsed,
                input_summary=f"chart_context_count={len(state.get('chart_contexts', []))}",
                error_message=state.get("error_message", ""),
            )
            return state

        records: list[dict[str, Any]] = []
        success_count = 0
        failed_count = 0
        for context in state.get("chart_contexts", []):
            record: dict[str, Any] = {}
            try:
                insight = self._workflow.insight_agent.generate_insight(context)
                success_count += 1
                record = {
                    **context,
                    "insight": insight,
                    "success": True,
                    "error_message": "",
                }
            except Exception as exc:
                failed_count += 1
                record = {
                    **context,
                    "insight": INSIGHT_FALLBACK,
                    "success": False,
                    "error_message": str(exc),
                }
            records.append(record)

        state["insights"] = records
        elapsed = _record_latency(state, "generate_insights", node_start)
        if failed_count == 0:
            status = "success"
        elif success_count > 0:
            status = "partial_success"
        else:
            status = "error"

        metadata = {
            "chart_context_count": len(state.get("chart_contexts", [])),
            "insight_success_count": success_count,
            "insight_failed_count": failed_count,
        }
        metadata.update(_collect_ollama_metrics(self._workflow.insight_agent))
        _append_tool_call(
            state=state,
            audit_logger=self.audit_logger,
            component="ollama",
            action="generate_insights",
            tool="ollama.generate_insights",
            status=status,
            latency_seconds=elapsed,
            input_summary=f"chart_context_count={len(state.get('chart_contexts', []))}",
            output_summary=(
                f"insight_success_count={success_count}; "
                f"insight_failed_count={failed_count}"
            ),
            metadata=metadata,
        )
        return state

    def build_report(self, state: ReportGraphState) -> ReportGraphState:
        return self._run_blocking_node(
            state=state,
            node_name="build_report",
            component="report",
            action="build_report",
            tool="report.build_report",
            func=lambda: self._workflow._build_report(list(state.get("insights", []))),
            output_key="report",
            input_summary=f"insight_count={len(state.get('insights', []))}",
        )

    def render_latex(self, state: ReportGraphState) -> ReportGraphState:
        return self._run_blocking_node(
            state=state,
            node_name="render_latex",
            component="report",
            action="render_latex",
            tool="report.render_latex",
            func=lambda: render_latex_report(
                report=state["report"],
                template_path=self._workflow.template_path,
                output_path=self._workflow.latex_output_path,
            ),
            output_key="tex_path",
            input_summary=f"output_path={self._workflow.latex_output_path}",
        )

    def compile_pdf(self, state: ReportGraphState) -> ReportGraphState:
        start = perf_counter()
        if _has_blocking_error(state) or not state.get("tex_path"):
            error_message = state.get("error_message", "")
            if not state.get("tex_path") and not error_message:
                error_message = "LaTeX output path is not available."
            elapsed = _record_latency(state, "compile_pdf", start)
            _append_tool_call(
                state=state,
                audit_logger=self.audit_logger,
                component="report",
                action="compile_pdf",
                tool="report.compile_pdf",
                status="skipped",
                latency_seconds=elapsed,
                input_summary=str(state.get("tex_path", "")),
                error_message=error_message,
            )
            return state

        try:
            pdf_path = compile_pdf(
                tex_path=state["tex_path"],
                output_dir=self._workflow.pdf_dir,
            )
            state["pdf_path"] = pdf_path
            status = "success"
            error_message = ""
            output_summary = str(pdf_path)
        except Exception as exc:
            state["pdf_error"] = str(exc)
            status = "error"
            error_message = str(exc)
            output_summary = ""
        finally:
            elapsed = _record_latency(state, "compile_pdf", start)
            _append_tool_call(
                state=state,
                audit_logger=self.audit_logger,
                component="report",
                action="compile_pdf",
                tool="report.compile_pdf",
                status=status,
                latency_seconds=elapsed,
                input_summary=str(state.get("tex_path", "")),
                output_summary=output_summary,
                error_message=error_message,
            )
        return state

    def write_report_log(self, state: ReportGraphState) -> ReportGraphState:
        start = perf_counter()
        metadata = self._build_metadata(state)
        try:
            log_path = self._workflow._write_log(metadata)
            state["log_path"] = log_path
            metadata["log_path"] = str(log_path)
            state["metadata"] = metadata
            status = "success"
            error_message = ""
            output_summary = str(log_path)
        except Exception as exc:
            state["error_message"] = str(exc)
            metadata = self._build_metadata(state)
            state["metadata"] = metadata
            status = "error"
            error_message = str(exc)
            output_summary = ""
        finally:
            elapsed = _record_latency(state, "write_report_log", start)
            _append_tool_call(
                state=state,
                audit_logger=self.audit_logger,
                component="report",
                action="write_log",
                tool="report.write_log",
                status=status,
                latency_seconds=elapsed,
                input_summary=str(self._workflow.log_path),
                output_summary=output_summary,
                error_message=error_message,
            )
        return state

    def finalize_report(self, state: ReportGraphState) -> ReportGraphState:
        start = perf_counter()
        _record_latency(state, "finalize_report", start)
        state["metadata"] = self._build_metadata(state)
        return state

    def _run_blocking_node(
        self,
        state: ReportGraphState,
        node_name: str,
        component: str,
        action: str,
        tool: str,
        func,
        output_key: str,
        input_summary: str = "",
    ) -> ReportGraphState:
        start = perf_counter()
        if _has_blocking_error(state):
            elapsed = _record_latency(state, node_name, start)
            _append_tool_call(
                state=state,
                audit_logger=self.audit_logger,
                component=component,
                action=action,
                tool=tool,
                status="skipped",
                latency_seconds=elapsed,
                input_summary=input_summary,
                error_message=state.get("error_message", ""),
            )
            return state

        status = "success"
        error_message = ""
        result: Any = None
        try:
            result = func()
            state[output_key] = result
        except Exception as exc:
            status = "error"
            error_message = str(exc)
            state["error_message"] = error_message
        finally:
            elapsed = _record_latency(state, node_name, start)
            _append_tool_call(
                state=state,
                audit_logger=self.audit_logger,
                component=component,
                action=action,
                tool=tool,
                status=status,
                latency_seconds=elapsed,
                input_summary=input_summary,
                output_summary=summarize_value(result),
                error_message=error_message,
            )
        return state

    def _build_metadata(self, state: ReportGraphState) -> dict[str, Any]:
        timestamp_end = state.get("timestamp_end") or _utc_now()
        state["timestamp_end"] = timestamp_end
        charts = list(state.get("charts", []))
        insights = list(state.get("insights", []))
        tex_path = state.get("tex_path")
        pdf_path = state.get("pdf_path")
        error_message = state.get("error_message", "")
        pdf_error = state.get("pdf_error", "")
        tex_success = tex_path is not None
        pdf_success = pdf_path is not None

        return {
            "engine": "langgraph",
            "timestamp_start": state.get("timestamp_start", ""),
            "timestamp_end": timestamp_end,
            "success": tex_success and pdf_success and not error_message and not pdf_error,
            "tex_success": tex_success,
            "pdf_success": pdf_success,
            "error_message": error_message,
            "pdf_error": pdf_error,
            "db_path": str(self._workflow.db_path),
            "figures_dir": str(self._workflow.figures_dir),
            "audit_log_path": str(self.audit_logger.log_path),
            "tex_path": str(tex_path) if tex_path else "",
            "pdf_path": str(pdf_path) if pdf_path else "",
            "log_path": str(self._workflow.log_path),
            "chart_count": len(charts),
            "insight_success_count": sum(
                1 for record in insights if record.get("success")
            ),
            "insight_failed_count": sum(
                1 for record in insights if not record.get("success")
            ),
            "latency": dict(state.get("latency", {})),
            "tool_calls": list(state.get("tool_calls", [])),
            "charts": charts,
            "insights": insights,
        }


def run_langgraph_report_workflow(domain: str = "energy") -> dict:
    """Run the default LangGraph report workflow."""
    workflow = LangGraphReportWorkflow(domain=domain)
    return workflow.run()


def _append_tool_call(
    state: ReportGraphState,
    audit_logger: ToolAuditLogger,
    component: str,
    action: str,
    tool: str,
    status: str,
    latency_seconds: float,
    input_summary: str = "",
    output_summary: str = "",
    error_message: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    event_metadata = metadata or {}
    timestamp = _utc_now()
    try:
        event = audit_logger.log(
            component=component,
            action=action,
            tool=tool,
            status=status,
            latency_seconds=latency_seconds,
            input_summary=input_summary,
            output_summary=output_summary,
            error_message=error_message,
            metadata=event_metadata,
            timestamp=timestamp,
        )
    except Exception:
        event = _build_tool_call_record(
            timestamp=timestamp,
            component=component,
            action=action,
            tool=tool,
            status=status,
            latency_seconds=latency_seconds,
            input_summary=input_summary,
            output_summary=output_summary,
            error_message=error_message,
            metadata=event_metadata,
        )

    if not isinstance(event, dict):
        event = _build_tool_call_record(
            timestamp=timestamp,
            component=component,
            action=action,
            tool=tool,
            status=status,
            latency_seconds=latency_seconds,
            input_summary=input_summary,
            output_summary=output_summary,
            error_message=error_message,
            metadata=event_metadata,
        )

    state.setdefault("tool_calls", []).append(event)


def _collect_ollama_metrics(agent: Any) -> dict[str, Any]:
    ollama_tool = getattr(agent, "ollama_tool", None)
    get_last_metrics = getattr(ollama_tool, "get_last_metrics", None)
    if not callable(get_last_metrics):
        return {}

    try:
        metrics = get_last_metrics()
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


def _record_latency(
    state: ReportGraphState,
    node_name: str,
    start: float,
) -> float:
    elapsed = perf_counter() - start
    state.setdefault("latency", {})[node_name] = elapsed
    return elapsed


def _has_blocking_error(state: ReportGraphState) -> bool:
    return bool(state.get("error_message"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
