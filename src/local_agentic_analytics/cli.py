"""Command line entry point for local-agentic-analytics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

from local_agentic_analytics.core.config import PROJECT_ROOT, load_config
from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.evaluation.logger import append_run_log
from local_agentic_analytics.graph.report_workflow import (
    DEFAULT_DB_PATH as DEFAULT_REPORT_DB_PATH,
    EnergyReportWorkflow,
)
from local_agentic_analytics.graph.workflow import SequentialAnalyticsWorkflow


MISSING_DATABASE_MESSAGE = (
    "Database belum ditemukan. Jalankan python scripts/ingest_energy.py terlebih dahulu."
)
QA_ENGINES = ("custom", "langgraph")
REPORT_ENGINES = ("custom", "langgraph")
_LANGGRAPH_WORKFLOW_RUNNER: Callable[..., AnalyticsState] | None = None
_LANGGRAPH_REPORT_WORKFLOW_RUNNER: Callable[[str], dict] | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local_agentic_analytics",
        description="CLI untuk local-agentic-analytics.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser(
        "ask",
        help="Jalankan mode Q&A text-to-SQL untuk dataset terpilih.",
    )
    ask_parser.add_argument(
        "question",
        nargs="+",
        help="Pertanyaan analitik untuk domain dataset.",
    )
    ask_parser.add_argument(
        "--domain",
        default="energy",
        help="Domain dataset untuk Q&A. Default: energy.",
    )
    ask_parser.add_argument(
        "--engine",
        choices=QA_ENGINES,
        default="custom",
        help="Orkestrator Q&A yang digunakan. Default: custom.",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="Generate laporan otomatis.",
    )
    report_parser.add_argument(
        "report_type",
        choices=["energy"],
        help="Jenis laporan yang akan dibuat.",
    )
    report_parser.add_argument(
        "--engine",
        choices=REPORT_ENGINES,
        default="custom",
        help="Orkestrator report yang digunakan. Default: custom.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ask":
        return run_ask(
            " ".join(args.question).strip(),
            engine=args.engine,
            domain=args.domain,
        )
    if args.command == "report" and args.report_type == "energy":
        return run_energy_report(engine=args.engine)

    parser.error("Unsupported command")
    return 2


def run_ask(user_query: str, engine: str = "custom", domain: str = "energy") -> int:
    if not user_query:
        print("Pertanyaan tidak boleh kosong.")
        return 1
    if engine not in QA_ENGINES:
        print(f"Engine tidak didukung: {engine}")
        return 1
    if not domain or not domain.strip():
        print("Domain tidak boleh kosong.")
        return 1

    domain = domain.strip()

    db_path = get_default_duckdb_path()
    if not db_path.exists():
        print(get_missing_database_message(domain))
        return 1

    state = run_qa_workflow(user_query, engine, domain=domain)
    append_run_log(state_to_run_log(state, engine=engine, domain=domain))
    print_qa_state(state)
    return 0 if state.success else 1


def run_qa_workflow(
    user_query: str,
    engine: str,
    domain: str = "energy",
) -> AnalyticsState:
    if engine == "langgraph":
        runner = _LANGGRAPH_WORKFLOW_RUNNER
        if runner is None:
            from local_agentic_analytics.graph.langgraph_workflow import (
                run_langgraph_workflow as runner,
            )

        return runner(user_query, domain=domain)

    workflow = SequentialAnalyticsWorkflow(domain=domain)
    return workflow.run(user_query)


def run_energy_report(engine: str = "custom") -> int:
    if engine not in REPORT_ENGINES:
        print(f"Engine report tidak didukung: {engine}")
        return 1

    db_path = DEFAULT_REPORT_DB_PATH
    if not db_path.exists():
        print(MISSING_DATABASE_MESSAGE)
        return 1

    metadata = run_report_workflow(engine=engine, db_path=db_path)
    metadata = normalize_report_metadata(metadata, engine=engine)
    print_report_metadata(metadata)

    return 0 if metadata.get("tex_success") else 1


def normalize_report_metadata(metadata: dict, engine: str) -> dict:
    normalized = dict(metadata)
    normalized.setdefault("engine", engine)
    normalized.setdefault("log_path", "")
    normalized.setdefault("latency", {})
    normalized.setdefault("tool_calls", [])
    return normalized


def run_report_workflow(engine: str, db_path: Path) -> dict:
    if engine == "langgraph":
        runner = _LANGGRAPH_REPORT_WORKFLOW_RUNNER
        if runner is None:
            from local_agentic_analytics.graph.langgraph_report_workflow import (
                run_langgraph_report_workflow as runner,
            )

        return runner("energy")

    workflow = EnergyReportWorkflow(db_path=db_path)
    return workflow.run()


def get_default_duckdb_path() -> Path:
    config = load_config("duckdb.yaml")
    db_path = config.get("duckdb", {}).get("database_path")
    if not isinstance(db_path, str) or not db_path.strip():
        return PROJECT_ROOT / "databases" / "duckdb" / "analytics.duckdb"

    path = Path(db_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_missing_database_message(domain: str) -> str:
    if domain == "energy":
        return MISSING_DATABASE_MESSAGE
    return (
        "Database belum ditemukan. Jalankan script ingestion untuk domain "
        f"'{domain}' terlebih dahulu, misalnya python scripts/ingest_{domain}.py "
        "jika tersedia."
    )


def print_qa_state(state: AnalyticsState) -> None:
    executed_sql = state.repaired_sql or state.generated_sql

    print("Generated SQL:")
    print(state.generated_sql or "-")

    if state.repaired_sql:
        print()
        print("Repaired SQL:")
        print(state.repaired_sql)

    print()
    print("Result:")
    print(json.dumps(state.sql_result or {}, ensure_ascii=False, indent=2))

    print()
    print("Final answer:")
    print(state.final_answer or "-")

    print()
    print("Latency:")
    for step_name, seconds in state.latency.items():
        print(f"- {step_name}: {seconds:.3f}s")

    print()
    print("Tool calls:")
    if state.tool_calls:
        for tool_call in state.tool_calls:
            tool = tool_call.get("tool", "-")
            status = tool_call.get("status", "-")
            latency = tool_call.get("latency_seconds", 0.0)
            try:
                latency_text = f"{float(latency):.3f}s"
            except (TypeError, ValueError):
                latency_text = "-"
            print(f"- {tool}: {status}, {latency_text}")
    else:
        print("-")

    print()
    print(f"Status: {'sukses' if state.success else 'gagal'}")
    if not state.success:
        print(f"Error: {state.error_message or 'Unknown error'}")
        if executed_sql:
            print()
            print("Last SQL:")
            print(executed_sql)


def print_report_metadata(metadata: dict) -> None:
    print("Energy report:")
    print(f"- engine: {metadata.get('engine') or 'custom'}")
    print(f"- success: {bool(metadata.get('success'))}")
    print(f"- tex_success: {bool(metadata.get('tex_success'))}")
    print(f"- pdf_success: {bool(metadata.get('pdf_success'))}")
    print(f"- tex_path: {metadata.get('tex_path') or '-'}")
    print(f"- pdf_path: {metadata.get('pdf_path') or '-'}")
    print(f"- log_path: {metadata.get('log_path') or '-'}")
    print(f"- chart_count: {metadata.get('chart_count', 0)}")

    print()
    print("Latency:")
    latency = metadata.get("latency")
    if isinstance(latency, dict):
        if latency:
            for step_name, seconds in latency.items():
                try:
                    latency_text = f"{float(seconds):.3f}s"
                except (TypeError, ValueError):
                    latency_text = "-"
                print(f"- {step_name}: {latency_text}")
        else:
            print("{}")
    else:
        print("-")

    print()
    print("Tool calls:")
    tool_calls = metadata.get("tool_calls")
    if isinstance(tool_calls, list):
        if tool_calls:
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                tool = tool_call.get("tool", "-")
                status = tool_call.get("status", "-")
                latency_seconds = tool_call.get("latency_seconds", 0.0)
                try:
                    latency_text = f"{float(latency_seconds):.3f}s"
                except (TypeError, ValueError):
                    latency_text = "-"
                print(f"- {tool}: {status}, {latency_text}")
        else:
            print("[]")
    else:
        print("-")

    error_message = metadata.get("error_message")
    pdf_error = metadata.get("pdf_error")
    if error_message:
        print()
        print(f"error_message: {error_message}")
    if pdf_error:
        print()
        print(f"pdf_error: {pdf_error}")


def state_to_run_log(
    state: AnalyticsState,
    engine: str = "custom",
    domain: str = "energy",
) -> dict:
    return {
        "engine": engine,
        "domain": domain,
        "user_query": state.user_query,
        "generated_sql": state.generated_sql or "",
        "repaired_sql": state.repaired_sql or "",
        "success": state.success,
        "error_message": state.error_message or "",
        "latency": state.latency,
        "selected_tools": state.selected_tools,
        "tool_calls": state.tool_calls,
        "route": state.route or "",
    }


if __name__ == "__main__":
    raise SystemExit(main())
