"""Hybrid finance query demo: DuckDB price summary + ChromaDB news, one answer.

This is the key demonstration that the two finance data sources are SEPARATE
yet connectable at the application layer:

  Step 1 (structured)   : SQL over DuckDB ``stock_prices`` -> price summary.
  Step 2 (unstructured) : ChromaDB ``finance_news`` retrieval filtered by ticker.
  Step 3 (fusion)       : both pieces handed to the ReporterAgent for one
                          cohesive narrative answer.

Run:
    python scripts/run_finance_hybrid_query.py NVDA --start 2019-06-01 --end 2019-06-30
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.reporter_agent import ReporterAgent
from local_agentic_analytics.core.config import load_config
from local_agentic_analytics.graph.workflow import dataframe_to_compact_result
from local_agentic_analytics.tools.chromadb_tool import (
    DEFAULT_PERSIST_DIRECTORY,
    ChromaDBTool,
)
from local_agentic_analytics.tools.duckdb_tool import DuckDBTool

DEFAULT_TABLE_NAME = "stock_prices"
DEFAULT_COLLECTION_NAME = "finance_news"


def _build_summary_sql(ticker: str, start: str, end: str, table_name: str) -> str:
    return (
        "SELECT\n"
        "    MIN(close) AS min_close_usd,\n"
        "    AVG(close) AS avg_close_usd,\n"
        "    MAX(close) AS max_close_usd,\n"
        "    COUNT(*) AS trading_days\n"
        f"FROM {table_name}\n"
        f"WHERE ticker = '{ticker}'\n"
        f"  AND CAST(date AS DATE) BETWEEN DATE '{start}' AND DATE '{end}';"
    )


def _headlines_from_matches(matches: list[dict[str, Any]]) -> list[dict[str, str]]:
    headlines: list[dict[str, str]] = []
    for match in matches:
        metadata = match.get("metadata") or {}
        headlines.append(
            {
                "headline": str(match.get("document", "")),
                "ticker": str(metadata.get("ticker", "")),
                "date": str(metadata.get("date", "")),
                "publisher": str(metadata.get("publisher", "")),
            }
        )
    return headlines


def run_hybrid_query(
    ticker: str,
    start: str,
    end: str,
    *,
    duckdb_tool: DuckDBTool,
    chroma_tool: ChromaDBTool,
    reporter: ReporterAgent,
    table_name: str = DEFAULT_TABLE_NAME,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run the structured + unstructured fusion for one ticker and period."""
    if not ticker or not ticker.strip():
        raise ValueError("ticker must not be empty")

    ticker = ticker.strip().upper()
    sql = _build_summary_sql(ticker, start, end, table_name)
    result_df = duckdb_tool.execute_query(sql)
    sql_result = dataframe_to_compact_result(result_df)

    matches = chroma_tool.query(
        text=f"berita {ticker} {start} {end}",
        top_k=top_k,
        where={"ticker": ticker},
    )
    retrieved_headlines = _headlines_from_matches(matches)

    question = (
        f"Ringkas pergerakan harga saham {ticker} pada periode {start} sampai "
        f"{end} dan kaitkan dengan berita yang tersedia."
    )
    answer = reporter.generate_answer(
        question=question,
        sql=sql,
        query_result={
            "price_summary": sql_result,
            "retrieved_headlines": retrieved_headlines,
        },
    )

    return {
        "ticker": ticker,
        "start": start,
        "end": end,
        "sql": sql,
        "sql_result": sql_result,
        "retrieved_headlines": retrieved_headlines,
        "answer": answer,
    }


def _default_db_path() -> Path:
    config = load_config("duckdb.yaml")
    db_path = config.get("duckdb", {}).get("database_path") if isinstance(config, dict) else None
    if not isinstance(db_path, str) or not db_path.strip():
        return PROJECT_ROOT / "databases" / "duckdb" / "analytics.duckdb"
    path = Path(db_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _build_chroma_tool(collection_name: str) -> ChromaDBTool:
    config = load_config("chromadb.yaml")
    chroma_config = config.get("chromadb", {}) if isinstance(config, dict) else {}
    embedding_config = config.get("embedding", {}) if isinstance(config, dict) else {}
    persist_directory = chroma_config.get(
        "persist_directory", str(DEFAULT_PERSIST_DIRECTORY)
    )
    persist_path = Path(persist_directory)
    if not persist_path.is_absolute():
        persist_path = PROJECT_ROOT / persist_path
    return ChromaDBTool(
        persist_directory=persist_path,
        collection_name=collection_name,
        embedding_model_name=str(
            embedding_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid finance query demo.")
    parser.add_argument("ticker", help="Ticker symbol (e.g. NVDA).")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of headlines.")
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="ChromaDB collection name.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    db_path = _default_db_path()
    if not db_path.exists():
        print(
            "Database belum ditemukan. Jalankan python scripts/ingest_finance_prices.py "
            "terlebih dahulu."
        )
        return 1

    duckdb_tool = DuckDBTool(str(db_path))
    chroma_tool = _build_chroma_tool(args.collection_name)

    result = run_hybrid_query(
        args.ticker,
        args.start,
        args.end,
        duckdb_tool=duckdb_tool,
        chroma_tool=chroma_tool,
        reporter=ReporterAgent(),
        top_k=args.top_k,
    )
    duckdb_tool.close()

    print(f"Ticker: {result['ticker']} | {result['start']} -> {result['end']}")
    print()
    print("SQL result (price summary):")
    print(result["sql_result"])
    print()
    print("Retrieved headlines:")
    if result["retrieved_headlines"]:
        for item in result["retrieved_headlines"]:
            print(
                f"- [{item['ticker']} | {item['date']} | {item['publisher']}] "
                f"{item['headline']}"
            )
    else:
        print("- (tidak ada berita yang cocok)")
    print()
    print("Final combined answer:")
    print(result["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
