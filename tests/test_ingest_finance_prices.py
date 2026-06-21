from pathlib import Path
import sys

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest_finance_prices import build_price_frame, ingest_finance_prices


def _fake_frame(base_close: float) -> pd.DataFrame:
    index = pd.to_datetime(["2019-01-02", "2019-01-03", "2019-01-04"])
    index.name = "Date"
    return pd.DataFrame(
        {
            "Open": [base_close - 1, base_close, base_close + 1],
            "High": [base_close + 2, base_close + 2, base_close + 3],
            "Low": [base_close - 2, base_close - 1, base_close],
            "Close": [base_close, base_close + 1, base_close + 2],
            "Adj Close": [base_close, base_close + 1, base_close + 2],
            "Volume": [1000000, 1100000, 1200000],
        },
        index=index,
    )


def _fake_downloader(ticker: str, start: str, end: str) -> pd.DataFrame:
    base = {"NVDA": 100.0, "TSLA": 200.0}[ticker]
    return _fake_frame(base)


def test_build_price_frame_produces_long_schema():
    frame = build_price_frame(
        ("NVDA", "TSLA"), "2019-01-01", "2019-02-01", _fake_downloader
    )

    assert list(frame.columns) == [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert set(frame["ticker"]) == {"NVDA", "TSLA"}
    assert len(frame) == 6


def test_ingest_finance_prices_creates_stock_prices_table(tmp_path):
    db_path = tmp_path / "analytics.duckdb"

    metadata = ingest_finance_prices(
        db_path=db_path,
        tickers=("NVDA", "TSLA"),
        start="2019-01-01",
        end="2019-02-01",
        downloader=_fake_downloader,
    )

    assert metadata["table_name"] == "stock_prices"
    assert metadata["row_count"] == 6
    assert metadata["ticker_counts"] == {"NVDA": 3, "TSLA": 3}

    with duckdb.connect(str(db_path)) as connection:
        columns = [
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(stock_prices)"
            ).fetchall()
        ]
        avg_close = connection.execute(
            "SELECT AVG(close) FROM stock_prices WHERE ticker = 'NVDA'"
        ).fetchone()[0]

    assert columns == ["date", "ticker", "open", "high", "low", "close", "volume"]
    assert round(avg_close, 2) == 101.0
