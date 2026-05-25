from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.insight_agent import InsightAgent
from local_agentic_analytics.visualization.chart_registry import ENERGY_CHART_REGISTRY
from local_agentic_analytics.visualization.chart_stats import CHART_STATS_REGISTRY


DEFAULT_DB_PATH = PROJECT_ROOT / "databases" / "duckdb" / "analytics.duckdb"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "experiments" / "chart_insights.json"


def build_chart_contexts(db_path: str | Path, figures_dir: str | Path) -> list[dict]:
    database_path = Path(db_path)
    if not database_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {database_path}")

    figure_root = Path(figures_dir)
    contexts = []
    con = duckdb.connect(str(database_path), read_only=True)
    try:
        for chart_id, chart_info in ENERGY_CHART_REGISTRY.items():
            stats_function = CHART_STATS_REGISTRY.get(chart_id)
            if stats_function is None:
                raise KeyError(f"No stats function registered for chart: {chart_id}")

            contexts.append(
                {
                    "chart_id": chart_id,
                    "chart_title": str(chart_info["title"]),
                    "chart_path": str(figure_root / str(chart_info["filename"])),
                    "stats": stats_function(con),
                }
            )
    finally:
        con.close()

    return contexts


def write_chart_insights(records: list[dict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main() -> int:
    try:
        contexts = build_chart_contexts(
            db_path=DEFAULT_DB_PATH,
            figures_dir=DEFAULT_FIGURES_DIR,
        )
    except Exception as exc:
        print(f"Gagal membuat statistik chart: {exc}")
        return 1

    agent = InsightAgent()
    timestamp = datetime.now(timezone.utc).isoformat()
    records = []
    for context in contexts:
        try:
            insight = agent.generate_insight(context)
            records.append(
                {
                    **context,
                    "insight": insight,
                    "success": True,
                    "error_message": "",
                    "timestamp": timestamp,
                }
            )
            print(f"Insight dibuat: {context['chart_id']}")
        except Exception as exc:
            records.append(
                {
                    **context,
                    "insight": "",
                    "success": False,
                    "error_message": str(exc),
                    "timestamp": timestamp,
                }
            )
            print(f"Gagal membuat insight {context['chart_id']}: {exc}")

    output_path = write_chart_insights(records, DEFAULT_OUTPUT_PATH)
    print(f"Output disimpan ke: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
