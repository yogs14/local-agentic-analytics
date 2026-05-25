from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.sql_cleaning import clean_sql_response


def test_clean_sql_response_removes_markdown_fence():
    sql = clean_sql_response(
        """
```sql
SELECT AVG(Global_active_power) AS avg_power
FROM electric_power;
```
        """
    )

    assert sql == "SELECT AVG(Global_active_power) AS avg_power\nFROM electric_power;"


def test_clean_sql_response_normalizes_date_filter():
    sql = clean_sql_response(
        "SELECT COUNT(*) FROM electric_power WHERE datetime = DATE '2006-12-16';"
    )

    assert "CAST(datetime AS DATE) = DATE '2006-12-16'" in sql


def test_clean_sql_response_adds_kwh_conversion_for_total_energy():
    sql = clean_sql_response(
        "SELECT SUM(Global_active_power) AS total_power FROM electric_power;",
        question="Berapa total energi kWh pada tanggal 16 Desember 2006?",
    )

    assert "SUM(Global_active_power) / 60.0 AS total_energy_kwh" in sql


def test_clean_sql_response_uses_filter_for_missing_value_count():
    sql = clean_sql_response(
        (
            "SELECT COUNT(DISTINCT CASE WHEN Global_active_power IS NULL THEN 1 END) "
            "FROM electric_power;"
        ),
        question="Berapa jumlah missing value pada kolom Global_active_power?",
    )

    assert sql == (
        "SELECT COUNT(*) FILTER (WHERE Global_active_power IS NULL) "
        "AS missing_global_active_power_count FROM electric_power;"
    )
