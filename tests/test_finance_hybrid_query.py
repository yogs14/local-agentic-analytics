from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_finance_hybrid_query import run_hybrid_query


class _FakeDuckDBTool:
    def __init__(self):
        self.executed_sql = []

    def execute_query(self, sql: str) -> pd.DataFrame:
        self.executed_sql.append(sql)
        return pd.DataFrame(
            {
                "min_close_usd": [100.0],
                "avg_close_usd": [110.0],
                "max_close_usd": [120.0],
                "trading_days": [20],
            }
        )


class _FakeChromaTool:
    def __init__(self):
        self.where = None

    def query(self, text: str, top_k: int = 3, where=None):
        self.where = where
        return [
            {
                "document": "NVDA hits new high",
                "metadata": {
                    "ticker": "NVDA",
                    "date": "2019-06-03",
                    "publisher": "Benzinga",
                },
                "distance": 0.1,
            }
        ]


class _FakeReporter:
    def __init__(self):
        self.payload = None

    def generate_answer(self, question: str, sql: str, query_result):
        self.payload = query_result
        return "Jawaban gabungan harga dan berita."


def test_run_hybrid_query_returns_sql_and_headlines():
    duckdb_tool = _FakeDuckDBTool()
    chroma_tool = _FakeChromaTool()
    reporter = _FakeReporter()

    result = run_hybrid_query(
        "nvda",
        "2019-06-01",
        "2019-06-30",
        duckdb_tool=duckdb_tool,
        chroma_tool=chroma_tool,
        reporter=reporter,
        top_k=3,
    )

    # Output exposes both structured and unstructured pieces.
    assert "sql_result" in result
    assert "retrieved_headlines" in result
    assert result["ticker"] == "NVDA"

    assert result["sql_result"]["rows"][0]["avg_close_usd"] == 110.0
    assert result["retrieved_headlines"][0]["ticker"] == "NVDA"
    assert result["retrieved_headlines"][0]["headline"] == "NVDA hits new high"
    assert result["answer"] == "Jawaban gabungan harga dan berita."

    # Ticker is upper-cased and used both in SQL and as the ChromaDB filter.
    assert "ticker = 'NVDA'" in duckdb_tool.executed_sql[0]
    assert chroma_tool.where == {"ticker": "NVDA"}

    # The reporter received both sources fused into one payload.
    assert "price_summary" in reporter.payload
    assert "retrieved_headlines" in reporter.payload
