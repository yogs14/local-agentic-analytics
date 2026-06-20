from pathlib import Path
import sys

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest_finance import ingest_finance_dataset


def test_ingest_finance_dataset_creates_stock_prices_table(tmp_path):
    db_path = tmp_path / "analytics.duckdb"

    metadata = ingest_finance_dataset(db_path)

    assert metadata["table_name"] == "stock_prices"
    assert metadata["row_count"] == 12

    with duckdb.connect(str(db_path)) as connection:
        columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(stock_prices)").fetchall()
        ]
        close_missing = connection.execute(
            "SELECT COUNT(*) FROM stock_prices WHERE close_price IS NULL"
        ).fetchone()[0]
        volume_missing = connection.execute(
            "SELECT COUNT(*) FROM stock_prices WHERE volume IS NULL"
        ).fetchone()[0]

    assert columns == ["date", "close_price", "volume", "return_pct"]
    assert close_missing == 1
    assert volume_missing == 1
