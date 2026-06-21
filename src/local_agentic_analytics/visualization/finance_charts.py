"""Deterministic finance charts built from DuckDB ``stock_prices``."""

from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


FINANCE_TABLE = "stock_prices"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TICKER_ORDER = ["NVDA", "NFLX", "TSLA", "GOOGL"]


def _ensure_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if not path.is_absolute():
        if path.parts[:2] == ("reports", "figures"):
            path = PROJECT_ROOT / path
        else:
            path = REPORTS_FIGURES_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _fetch_dataframe(con: Any, sql: str) -> pd.DataFrame:
    try:
        return con.execute(sql).fetchdf()
    except Exception as exc:
        raise ValueError(f"Failed to run visualization query: {exc}") from exc


def _save_figure(output_path: str | Path) -> Path:
    path = _ensure_output_path(output_path)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _ordered_tickers(values: Any) -> list[str]:
    present = list(dict.fromkeys(str(value) for value in values))
    ordered = [ticker for ticker in TICKER_ORDER if ticker in present]
    ordered.extend(ticker for ticker in present if ticker not in ordered)
    return ordered


def plot_close_price_trend(con: Any, output_path: str | Path) -> Path:
    query = f"""
        SELECT date, ticker, close
        FROM {FINANCE_TABLE}
        WHERE close IS NOT NULL
        ORDER BY ticker, date
    """
    df = _fetch_dataframe(con, query)
    if df.empty:
        raise ValueError("No close price data available for plotting.")

    df["date"] = pd.to_datetime(df["date"])
    plt.figure(figsize=(10, 4.8))
    for ticker in _ordered_tickers(df["ticker"]):
        subset = df[df["ticker"] == ticker]
        plt.plot(subset["date"], subset["close"], linewidth=1.3, label=ticker)
    plt.title("Tren Harga Penutupan Harian per Ticker")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga penutupan (USD)")
    plt.legend(title="Ticker")
    plt.grid(True, alpha=0.3)
    return _save_figure(output_path)


def plot_average_close_by_ticker(con: Any, output_path: str | Path) -> Path:
    query = f"""
        SELECT ticker, AVG(close) AS avg_close_usd
        FROM {FINANCE_TABLE}
        WHERE close IS NOT NULL
        GROUP BY ticker
    """
    df = _fetch_dataframe(con, query)
    if df.empty:
        raise ValueError("No close price data available for plotting.")

    df = df.set_index("ticker").reindex(_ordered_tickers(df["ticker"])).reset_index()
    plt.figure(figsize=(7.5, 4.8))
    plt.bar(df["ticker"], df["avg_close_usd"])
    plt.title("Rata-rata Harga Penutupan per Ticker")
    plt.xlabel("Ticker")
    plt.ylabel("Rata-rata harga penutupan (USD)")
    plt.grid(axis="y", alpha=0.3)
    return _save_figure(output_path)


def plot_average_volume_by_ticker(con: Any, output_path: str | Path) -> Path:
    query = f"""
        SELECT ticker, AVG(volume) AS avg_volume_shares
        FROM {FINANCE_TABLE}
        WHERE volume IS NOT NULL
        GROUP BY ticker
    """
    df = _fetch_dataframe(con, query)
    if df.empty:
        raise ValueError("No volume data available for plotting.")

    df = df.set_index("ticker").reindex(_ordered_tickers(df["ticker"])).reset_index()
    plt.figure(figsize=(7.5, 4.8))
    plt.bar(df["ticker"], df["avg_volume_shares"], color="#5B8FF9")
    plt.title("Rata-rata Volume Perdagangan per Ticker")
    plt.xlabel("Ticker")
    plt.ylabel("Rata-rata volume (lembar)")
    plt.grid(axis="y", alpha=0.3)
    return _save_figure(output_path)


def plot_daily_return_distribution(con: Any, output_path: str | Path) -> Path:
    query = f"""
        WITH returns AS (
            SELECT
                100.0 * (close - LAG(close) OVER (PARTITION BY ticker ORDER BY date))
                    / LAG(close) OVER (PARTITION BY ticker ORDER BY date)
                    AS return_pct
            FROM {FINANCE_TABLE}
            WHERE close IS NOT NULL
        )
        SELECT return_pct
        FROM returns
        WHERE return_pct IS NOT NULL
    """
    df = _fetch_dataframe(con, query)
    if df.empty:
        raise ValueError("No daily return data available for plotting.")

    plt.figure(figsize=(10, 4.8))
    plt.hist(df["return_pct"], bins=60, color="#61A0A8")
    plt.title("Distribusi Return Harian (Semua Ticker)")
    plt.xlabel("Return harian (%)")
    plt.ylabel("Jumlah hari perdagangan")
    plt.grid(axis="y", alpha=0.3)
    return _save_figure(output_path)


def plot_ticker_close_correlation(con: Any, output_path: str | Path) -> Path:
    query = f"""
        SELECT date, ticker, close
        FROM {FINANCE_TABLE}
        WHERE close IS NOT NULL
    """
    df = _fetch_dataframe(con, query)
    if df.empty:
        raise ValueError("No close price data available for plotting.")

    pivot = df.pivot_table(index="date", columns="ticker", values="close")
    tickers = _ordered_tickers(pivot.columns)
    pivot = pivot[tickers]
    corr_df = pivot.corr()
    if corr_df.isna().all().all():
        raise ValueError("No correlation data available for plotting.")

    plt.figure(figsize=(6.5, 5.5))
    image = plt.imshow(corr_df, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04, label="Korelasi")
    plt.title("Korelasi Harga Penutupan Antar Ticker")
    plt.xticks(range(len(tickers)), tickers, rotation=35, ha="right")
    plt.yticks(range(len(tickers)), tickers)
    for row_idx in range(len(tickers)):
        for col_idx in range(len(tickers)):
            value = corr_df.iloc[row_idx, col_idx]
            if pd.notna(value):
                plt.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
    return _save_figure(output_path)
