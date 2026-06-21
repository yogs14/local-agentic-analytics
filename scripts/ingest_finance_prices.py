"""Ingest daily stock prices (yfinance) into the DuckDB stock_prices table.

Structured (DuckDB) side of the hybrid finance domain. Downloads OHLCV data
for a fixed set of tickers and date range, then writes a single long-format
table ``stock_prices`` into the SAME analytics database used by energy
(separate table, separate concern).

The actual yfinance download is injected via ``downloader`` so the ingestion
logic can be unit-tested without any network access.

Run (downloads from the internet):
    python scripts/ingest_finance_prices.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "databases" / "duckdb" / "analytics.duckdb"
DEFAULT_TABLE_NAME = "stock_prices"
DEFAULT_TICKERS = ("NVDA", "NFLX", "TSLA", "GOOGL")
DEFAULT_START = "2019-01-01"
DEFAULT_END = "2020-06-10"

# yfinance OHLCV column name -> normalized table column name.
_OHLCV_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}

Downloader = Callable[[str, str, str], pd.DataFrame]


def quote_identifier(identifier: str) -> str:
    if not identifier or not identifier.strip():
        raise ValueError("Identifier must not be empty")

    return '"' + identifier.replace('"', '""') + '"'


def _default_downloader(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download one ticker's OHLCV history with yfinance (lazy import)."""
    import yfinance as yf

    return yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )


def _normalize_frame(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a single-ticker yfinance frame to the long table schema."""
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "volume"]
        )

    working = frame.copy()

    # yfinance may return a MultiIndex column frame when a list of tickers is
    # passed. Flatten by taking the first level (the OHLCV name).
    if isinstance(working.columns, pd.MultiIndex):
        working.columns = working.columns.get_level_values(0)

    working = working.reset_index()
    date_column = "Date" if "Date" in working.columns else working.columns[0]

    records = pd.DataFrame()
    records["date"] = pd.to_datetime(working[date_column]).dt.date
    records["ticker"] = ticker
    for source_name, target_name in _OHLCV_COLUMNS.items():
        if source_name not in working.columns:
            raise ValueError(
                f"Downloaded frame for {ticker} is missing column '{source_name}'"
            )
        records[target_name] = working[source_name].values

    return records[
        ["date", "ticker", "open", "high", "low", "close", "volume"]
    ]


def build_price_frame(
    tickers: tuple[str, ...],
    start: str,
    end: str,
    downloader: Downloader,
) -> pd.DataFrame:
    """Build the combined long-format price frame for all tickers."""
    if not tickers:
        raise ValueError("tickers must not be empty")

    frames = [
        _normalize_frame(ticker, downloader(ticker, start, end))
        for ticker in tickers
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["close"])
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    return combined


def ingest_finance_prices(
    db_path: Path,
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    table_name: str = DEFAULT_TABLE_NAME,
    downloader: Downloader | None = None,
) -> dict[str, Any]:
    """Create or replace ``stock_prices`` from downloaded OHLCV data."""
    downloader = downloader or _default_downloader
    price_frame = build_price_frame(tickers, start, end, downloader)
    if price_frame.empty:
        raise ValueError("No price rows were produced; check tickers and date range")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    table_identifier = quote_identifier(table_name)

    with duckdb.connect(str(db_path)) as connection:
        connection.execute("PRAGMA threads=1")
        connection.register("price_frame", price_frame)
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE {table_identifier} AS
            SELECT
                CAST(date AS DATE) AS date,
                CAST(ticker AS VARCHAR) AS ticker,
                CAST(open AS DOUBLE) AS open,
                CAST(high AS DOUBLE) AS high,
                CAST(low AS DOUBLE) AS low,
                CAST(close AS DOUBLE) AS close,
                CAST(volume AS BIGINT) AS volume
            FROM price_frame
            """
        )
        connection.unregister("price_frame")

        row_count = connection.execute(
            f"SELECT COUNT(*) FROM {table_identifier}"
        ).fetchone()[0]
        ticker_counts = connection.execute(
            f"SELECT ticker, COUNT(*) FROM {table_identifier} GROUP BY ticker "
            "ORDER BY ticker"
        ).fetchall()
        schema_rows = connection.execute(f"DESCRIBE {table_identifier}").fetchall()

    return {
        "db_path": str(db_path),
        "table_name": table_name,
        "row_count": int(row_count),
        "ticker_counts": {ticker: int(count) for ticker, count in ticker_counts},
        "schema": [
            {"column_name": row[0], "column_type": row[1]} for row in schema_rows
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest yfinance daily stock prices into DuckDB."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="DuckDB database path.",
    )
    parser.add_argument(
        "--table-name",
        default=DEFAULT_TABLE_NAME,
        help="Destination DuckDB table name.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_TICKERS),
        help="Ticker symbols to download.",
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (inclusive).")
    parser.add_argument("--end", default=DEFAULT_END, help="End date (exclusive).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        metadata = ingest_finance_prices(
            db_path=args.db_path,
            tickers=tuple(args.tickers),
            start=args.start,
            end=args.end,
            table_name=args.table_name,
        )
    except (duckdb.Error, ValueError, ImportError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"DuckDB database: {metadata['db_path']}")
    print(f"Table: {metadata['table_name']}")
    print(f"Rows: {metadata['row_count']}")
    print("Rows per ticker:")
    for ticker, count in metadata["ticker_counts"].items():
        print(f"- {ticker}: {count}")
    print("Schema:")
    for column in metadata["schema"]:
        print(f"- {column['column_name']}: {column['column_type']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
