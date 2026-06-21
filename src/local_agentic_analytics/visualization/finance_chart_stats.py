"""Compact DuckDB statistics for finance chart insights."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
import math
from typing import Any


FINANCE_TABLE = "stock_prices"
TICKER_ORDER = ["NVDA", "NFLX", "TSLA", "GOOGL"]


def _to_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)
    return value


def _fetch_all(con: Any, sql: str) -> list[dict[str, Any]]:
    try:
        cursor = con.execute(sql)
        rows = cursor.fetchall()
    except Exception as exc:
        raise ValueError(f"Failed to run chart stats query: {exc}") from exc

    columns = [description[0] for description in cursor.description]
    return [
        {column: _to_json_value(value) for column, value in zip(columns, row)}
        for row in rows
    ]


def _fetch_one(con: Any, sql: str) -> dict[str, Any]:
    rows = _fetch_all(con, sql)
    return rows[0] if rows else {}


def get_close_price_trend_stats(con: Any) -> dict:
    rows = _fetch_all(
        con,
        f"""
        SELECT
            ticker,
            COUNT(*) AS trading_days,
            CAST(MIN(date) AS VARCHAR) AS first_date,
            CAST(MAX(date) AS VARCHAR) AS last_date,
            arg_min(close, date) AS first_close,
            arg_max(close, date) AS last_close,
            MIN(close) AS min_close,
            MAX(close) AS max_close
        FROM {FINANCE_TABLE}
        WHERE close IS NOT NULL
        GROUP BY ticker
        """,
    )
    for row in rows:
        first_close = row.get("first_close")
        last_close = row.get("last_close")
        if isinstance(first_close, (int, float)) and first_close:
            row["change_pct"] = round(
                100.0 * (last_close - first_close) / first_close, 4
            )
        else:
            row["change_pct"] = None
    rows.sort(key=lambda item: _ticker_rank(item.get("ticker")))
    return {
        "unit": "USD",
        "ticker_count": len(rows),
        "per_ticker": rows,
    }


def get_average_close_by_ticker_stats(con: Any) -> dict:
    rows = _fetch_all(
        con,
        f"""
        SELECT ticker, AVG(close) AS avg_close_usd, COUNT(*) AS trading_days
        FROM {FINANCE_TABLE}
        WHERE close IS NOT NULL
        GROUP BY ticker
        """,
    )
    rows.sort(key=lambda item: _ticker_rank(item.get("ticker")))
    valid = [r for r in rows if isinstance(r.get("avg_close_usd"), (int, float))]
    return {
        "unit": "USD",
        "per_ticker": rows,
        "highest_ticker": (
            max(valid, key=lambda r: r["avg_close_usd"])["ticker"] if valid else None
        ),
        "lowest_ticker": (
            min(valid, key=lambda r: r["avg_close_usd"])["ticker"] if valid else None
        ),
    }


def get_average_volume_by_ticker_stats(con: Any) -> dict:
    rows = _fetch_all(
        con,
        f"""
        SELECT ticker, AVG(volume) AS avg_volume_shares
        FROM {FINANCE_TABLE}
        WHERE volume IS NOT NULL
        GROUP BY ticker
        """,
    )
    rows.sort(key=lambda item: _ticker_rank(item.get("ticker")))
    valid = [r for r in rows if isinstance(r.get("avg_volume_shares"), (int, float))]
    return {
        "unit": "shares",
        "per_ticker": rows,
        "highest_volume_ticker": (
            max(valid, key=lambda r: r["avg_volume_shares"])["ticker"]
            if valid
            else None
        ),
    }


def get_daily_return_distribution_stats(con: Any) -> dict:
    stats = _fetch_one(
        con,
        f"""
        WITH returns AS (
            SELECT
                ticker,
                100.0 * (close - LAG(close) OVER (PARTITION BY ticker ORDER BY date))
                    / LAG(close) OVER (PARTITION BY ticker ORDER BY date)
                    AS return_pct
            FROM {FINANCE_TABLE}
            WHERE close IS NOT NULL
        )
        SELECT
            COUNT(*) AS return_count,
            AVG(return_pct) AS mean_return_pct,
            stddev_samp(return_pct) AS stddev_return_pct,
            MIN(return_pct) AS min_return_pct,
            MAX(return_pct) AS max_return_pct
        FROM returns
        WHERE return_pct IS NOT NULL
        """,
    )
    stats["unit"] = "percent"
    return stats


def get_ticker_close_correlation_stats(con: Any) -> dict:
    rows = _fetch_all(
        con,
        f"""
        WITH paired AS (
            SELECT a.ticker AS ticker_a, b.ticker AS ticker_b,
                   corr(a.close, b.close) AS correlation
            FROM {FINANCE_TABLE} a
            JOIN {FINANCE_TABLE} b
              ON a.date = b.date AND a.ticker < b.ticker
            WHERE a.close IS NOT NULL AND b.close IS NOT NULL
            GROUP BY a.ticker, b.ticker
        )
        SELECT ticker_a, ticker_b, correlation FROM paired
        """,
    )
    pairs = [
        {
            "pair": f"{row['ticker_a']} vs {row['ticker_b']}",
            "correlation": row["correlation"],
        }
        for row in rows
        if isinstance(row.get("correlation"), (int, float))
    ]
    strongest_positive = max(pairs, key=lambda p: p["correlation"]) if pairs else None
    strongest_absolute = (
        max(pairs, key=lambda p: abs(p["correlation"])) if pairs else None
    )
    return {
        "pair_count": len(pairs),
        "correlations": pairs,
        "strongest_positive": strongest_positive,
        "strongest_absolute": strongest_absolute,
        "unit_note": "Korelasi tidak memiliki satuan; harga penutupan memakai USD.",
    }


def _ticker_rank(ticker: Any) -> tuple[int, str]:
    label = str(ticker)
    if label in TICKER_ORDER:
        return (TICKER_ORDER.index(label), label)
    return (len(TICKER_ORDER), label)


FINANCE_CHART_STATS_REGISTRY = {
    "close_price_trend": get_close_price_trend_stats,
    "average_close_by_ticker": get_average_close_by_ticker_stats,
    "average_volume_by_ticker": get_average_volume_by_ticker_stats,
    "daily_return_distribution": get_daily_return_distribution_stats,
    "ticker_close_correlation": get_ticker_close_correlation_stats,
}
