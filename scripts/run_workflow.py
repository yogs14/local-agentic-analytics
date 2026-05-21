"""Run the sequential analytics workflow from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.evaluation.logger import append_run_log
from local_agentic_analytics.graph.workflow import SequentialAnalyticsWorkflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local sequential text-to-SQL analytics workflow."
    )
    parser.add_argument(
        "question",
        nargs="+",
        help="User question to answer from the electric_power table.",
    )
    return parser.parse_args()


def print_state(state: AnalyticsState) -> None:
    executed_sql = state.repaired_sql or state.generated_sql

    print("SQL yang dihasilkan:")
    print(state.generated_sql or "-")

    if state.repaired_sql:
        print()
        print("SQL hasil repair:")
        print(state.repaired_sql)

    print()
    print("Hasil query:")
    print(json.dumps(state.sql_result or {}, ensure_ascii=False, indent=2))

    print()
    print("Jawaban akhir:")
    print(state.final_answer or "-")

    print()
    print("Latency:")
    for step_name, seconds in state.latency.items():
        print(f"- {step_name}: {seconds:.3f}s")

    if not state.success:
        print()
        print("Status: gagal")
        print(f"Error: {state.error_message or 'Unknown error'}")
        if executed_sql:
            print()
            print("SQL terakhir:")
            print(executed_sql)
    else:
        print()
        print("Status: sukses")


def state_to_run_log(state: AnalyticsState) -> dict:
    return {
        "user_query": state.user_query,
        "generated_sql": state.generated_sql or "",
        "repaired_sql": state.repaired_sql or "",
        "success": state.success,
        "error_message": state.error_message or "",
        "latency": state.latency,
    }


def main() -> int:
    args = parse_args()
    user_query = " ".join(args.question).strip()

    workflow = SequentialAnalyticsWorkflow()
    state = workflow.run(user_query)
    append_run_log(state_to_run_log(state))
    print_state(state)

    return 0 if state.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
