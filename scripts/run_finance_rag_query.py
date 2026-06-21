"""Finance RAG query demo over the ChromaDB ``finance_news`` collection.

Retrieves the most relevant news headlines for a free-text query and asks the
ReporterAgent to produce a short narrative answer grounded in them. This
exercises the UNSTRUCTURED (ChromaDB) path of the finance domain in isolation.

Run:
    python scripts/run_finance_rag_query.py "berita terbaru tentang NVDA"
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
from local_agentic_analytics.tools.chromadb_tool import (
    DEFAULT_PERSIST_DIRECTORY,
    ChromaDBTool,
)
from local_agentic_analytics.core.config import load_config

DEFAULT_COLLECTION_NAME = "finance_news"
NO_SQL_MARKER = "-- RAG retrieval finance_news (tanpa SQL)"


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


def run_rag_query(
    query: str,
    *,
    chroma_tool: ChromaDBTool,
    reporter: ReporterAgent,
    top_k: int = 5,
) -> dict[str, Any]:
    """Retrieve headlines for ``query`` and generate a narrative answer."""
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    matches = chroma_tool.query(text=query, top_k=top_k)
    retrieved_headlines = _headlines_from_matches(matches)

    answer = reporter.generate_answer(
        question=query,
        sql=NO_SQL_MARKER,
        query_result={"retrieved_headlines": retrieved_headlines},
    )

    return {
        "query": query,
        "retrieved_headlines": retrieved_headlines,
        "answer": answer,
    }


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
            embedding_config.get(
                "model", "sentence-transformers/all-MiniLM-L6-v2"
            )
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finance news RAG query demo.")
    parser.add_argument("query", nargs="+", help="Free-text news query.")
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="ChromaDB collection name.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of headlines.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query = " ".join(args.query).strip()

    chroma_tool = _build_chroma_tool(args.collection_name)
    if chroma_tool.count() == 0:
        print(
            f"Collection '{args.collection_name}' is empty. Jalankan "
            "python scripts/ingest_finance_news.py terlebih dahulu."
        )
        return 1

    result = run_rag_query(
        query,
        chroma_tool=chroma_tool,
        reporter=ReporterAgent(),
        top_k=args.top_k,
    )

    print(f"Query: {result['query']}")
    print()
    print("Retrieved headlines:")
    for item in result["retrieved_headlines"]:
        print(
            f"- [{item['ticker']} | {item['date']} | {item['publisher']}] "
            f"{item['headline']}"
        )
    print()
    print("Final answer:")
    print(result["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
