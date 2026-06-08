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
_LANGGRAPH_WORKFLOW_RUNNER: Callable[[str], AnalyticsState] | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local_agentic_analytics",
        description="CLI untuk local-agentic-analytics.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser(
        "ask",
        help="Jalankan mode Q&A text-to-SQL untuk dataset energi.",
    )
    ask_parser.add_argument(
        "question",
        nargs="+",
        help="Pertanyaan analitik untuk tabel electric_power.",
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ask":
        return run_ask(" ".join(args.question).strip(), engine=args.engine)
    if args.command == "report" and args.report_type == "energy":
        return run_energy_report()

    parser.error("Unsupported command")
    return 2


def run_ask(user_query: str, engine: str = "custom") -> int:
    if not user_query:
        print("Pertanyaan tidak boleh kosong.")
        return 1
    if engine not in QA_ENGINES:
        print(f"Engine tidak didukung: {engine}")
        return 1

    db_path = get_default_duckdb_path()
    if not db_path.exists():
        print(MISSING_DATABASE_MESSAGE)
        return 1

    state = run_qa_workflow(user_query, engine)
    append_run_log(state_to_run_log(state, engine=engine))
    print_qa_state(state)
    return 0 if state.success else 1


def run_qa_workflow(user_query: str, engine: str) -> AnalyticsState:
    if engine == "langgraph":
        runner = _LANGGRAPH_WORKFLOW_RUNNER
        if runner is None:
            from local_agentic_analytics.graph.langgraph_workflow import (
                run_langgraph_workflow as runner,
            )

        return runner(user_query)

    workflow = SequentialAnalyticsWorkflow()
    return workflow.run(user_query)


def run_energy_report() -> int:
    db_path = DEFAULT_REPORT_DB_PATH
    if not db_path.exists():
        print(MISSING_DATABASE_MESSAGE)
        return 1

    workflow = EnergyReportWorkflow(db_path=db_path)
    metadata = workflow.run()
    print_report_metadata(metadata)

    return 0 if metadata.get("tex_success") else 1


def get_default_duckdb_path() -> Path:
    config = load_config("duckdb.yaml")
    db_path = config.get("duckdb", {}).get("database_path")
    if not isinstance(db_path, str) or not db_path.strip():
        return PROJECT_ROOT / "databases" / "duckdb" / "analytics.duckdb"

    path = Path(db_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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
    pdf_success = bool(metadata.get("pdf_success"))
    compile_status = "sukses" if pdf_success else "gagal"

    print("Energy report:")
    print(f"- LaTeX path: {metadata.get('tex_path') or '-'}")
    print(f"- PDF path: {metadata.get('pdf_path') or '-'}")
    print(f"- Jumlah chart: {metadata.get('chart_count', 0)}")
    print(f"- Status compile: {compile_status}")

    error_message = metadata.get("error_message") or metadata.get("pdf_error")
    if error_message:
        print(f"- Error: {error_message}")


def state_to_run_log(state: AnalyticsState, engine: str = "custom") -> dict:
    return {
        "engine": engine,
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
