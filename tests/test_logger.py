from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation import logger


def test_append_run_log_creates_csv_with_expected_columns(tmp_path, monkeypatch):
    runs_csv = tmp_path / "reports" / "experiments" / "runs.csv"
    monkeypatch.setattr(logger, "RUNS_CSV_PATH", runs_csv)

    logger.append_run_log(
        {
            "timestamp": "2026-05-21T00:00:00+00:00",
            "user_query": "Berapa rata-rata konsumsi?",
            "generated_sql": "SELECT AVG(Global_active_power) FROM electric_power",
            "repaired_sql": "",
            "success": True,
            "error_message": "",
            "latency": {
                "total": 1.5,
                "sql_generation": 1.0,
                "sql_execution": 0.2,
                "reporting": 0.3,
            },
        }
    )

    lines = runs_csv.read_text(encoding="utf-8").splitlines()

    assert lines[0] == ",".join(logger.RUN_LOG_COLUMNS)
    assert "Berapa rata-rata konsumsi?" in lines[1]
    assert "1.5,1.0,0.2,0.3" in lines[1]


def test_append_run_log_appends_without_rewriting_header(tmp_path, monkeypatch):
    runs_csv = tmp_path / "reports" / "experiments" / "runs.csv"
    monkeypatch.setattr(logger, "RUNS_CSV_PATH", runs_csv)

    logger.append_run_log({"user_query": "query pertama"})
    logger.append_run_log({"user_query": "query kedua"})

    lines = runs_csv.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 3
    assert lines[0] == ",".join(logger.RUN_LOG_COLUMNS)
    assert "query pertama" in lines[1]
    assert "query kedua" in lines[2]
