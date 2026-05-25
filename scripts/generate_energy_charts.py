from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.visualization.chart_registry import generate_all_energy_charts


DEFAULT_DB_PATH = PROJECT_ROOT / "databases" / "duckdb" / "analytics.duckdb"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"


def main() -> int:
    try:
        chart_metadata = generate_all_energy_charts(
            db_path=DEFAULT_DB_PATH,
            output_dir=DEFAULT_OUTPUT_DIR,
        )
    except Exception as exc:
        print(f"Gagal membuat grafik energi: {exc}")
        return 1

    print("Grafik energi berhasil dibuat:")
    for chart in chart_metadata:
        print(f"- {chart['chart_id']}: {chart['path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
