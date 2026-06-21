"""Ingest financial news headlines into the ChromaDB ``finance_news`` collection.

Unstructured (ChromaDB) side of the hybrid finance domain. Reads a local CSV of
analyst-rating headlines, filters to the target tickers and date range, cleans
the timestamp down to a plain ``YYYY-MM-DD`` date, and embeds each headline into
a SEPARATE ChromaDB collection (``finance_news``) using the same local
sentence-transformers model as the rest of the project.

The download/embedding model and batch size come from ``configs/chromadb.yaml``;
only the collection name differs from the default RAG collection.

Run:
    python scripts/ingest_finance_news.py
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from local_agentic_analytics.core.config import load_config
from local_agentic_analytics.tools.chromadb_tool import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_PERSIST_DIRECTORY,
    ChromaDBTool,
)


DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "finance" / "raw_analyst_ratings.csv"
DEFAULT_COLLECTION_NAME = "finance_news"
DEFAULT_TICKERS = ("NVDA", "NFLX", "TSLA", "GOOGL")
DEFAULT_START = date(2019, 1, 1)
DEFAULT_END = date(2020, 6, 10)


def _coerce_date(value: str) -> date:
    if isinstance(value, str) and value.strip():
        return date(*(int(part) for part in value.strip().split("-")[:3]))
    raise ValueError(f"Invalid date bound: {value!r}")


def load_news_documents(
    csv_path: Path,
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
) -> dict[str, Any]:
    """Read and filter the news CSV into ChromaDB-ready documents/metadata.

    Returns a dict with ``documents``, ``metadatas`` and ``stats``. No network,
    no embedding -- this is the pure, unit-testable part of ingestion.
    """
    if not Path(csv_path).is_file():
        raise FileNotFoundError(f"News CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    total_rows = int(len(frame))

    required = {"headline", "url", "publisher", "date", "stock"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"News CSV missing columns: {sorted(missing)}")

    ticker_set = {ticker.upper() for ticker in tickers}
    frame = frame[frame["stock"].astype(str).str.upper().isin(ticker_set)].copy()

    # Strip timezone and keep the calendar date only (YYYY-MM-DD).
    parsed = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["clean_date"] = parsed.dt.date
    frame = frame.dropna(subset=["clean_date", "headline"])
    frame = frame[(frame["clean_date"] >= start) & (frame["clean_date"] <= end)]
    frame = frame[frame["headline"].astype(str).str.strip() != ""]

    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    for _, row in frame.iterrows():
        documents.append(str(row["headline"]).strip())
        metadatas.append(
            {
                "ticker": str(row["stock"]).upper(),
                "date": row["clean_date"].isoformat(),
                "publisher": _clean_optional(row.get("publisher")),
                "url": _clean_optional(row.get("url")),
            }
        )

    return {
        "documents": documents,
        "metadatas": metadatas,
        "stats": {
            "total_rows": total_rows,
            "matched_rows": len(documents),
        },
    }


def _clean_optional(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def embed_news(
    tool: ChromaDBTool,
    documents: list[str],
    metadatas: list[dict[str, str]],
    batch_size: int,
) -> dict[str, int]:
    """Embed documents into ChromaDB in batches, returning add/skip counts."""
    if batch_size < 1:
        raise ValueError("batch_size must be greater than 0")

    added = 0
    skipped = 0
    for start_index in range(0, len(documents), batch_size):
        doc_batch = documents[start_index : start_index + batch_size]
        meta_batch = metadatas[start_index : start_index + batch_size]
        if not doc_batch:
            continue
        summary = tool.add_documents(doc_batch, meta_batch)
        added += int(summary["added"])
        skipped += int(summary["skipped"])

    return {"added": added, "skipped": skipped}


def _build_chromadb_tool(collection_name: str) -> tuple[ChromaDBTool, int]:
    config = load_config("chromadb.yaml")
    chroma_config = config.get("chromadb", {}) if isinstance(config, dict) else {}
    embedding_config = config.get("embedding", {}) if isinstance(config, dict) else {}

    persist_directory = _resolve_project_path(
        str(chroma_config.get("persist_directory", DEFAULT_PERSIST_DIRECTORY))
    )
    model_name = str(embedding_config.get("model", DEFAULT_EMBEDDING_MODEL))
    batch_size = int(embedding_config.get("batch_size", 8))
    anonymized_telemetry = bool(chroma_config.get("anonymized_telemetry", False))

    tool = ChromaDBTool(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model_name=model_name,
        anonymized_telemetry=anonymized_telemetry,
    )
    return tool, batch_size


def _resolve_project_path(path: str) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest financial news headlines into ChromaDB finance_news."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to raw_analyst_ratings.csv.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Destination ChromaDB collection name.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        loaded = load_news_documents(args.csv_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    documents = loaded["documents"]
    metadatas = loaded["metadatas"]

    if not documents:
        print("No matching news rows found for the configured tickers/date range.")
        return 1

    tool, batch_size = _build_chromadb_tool(args.collection_name)
    summary = embed_news(tool, documents, metadatas, batch_size)

    print(f"Persist directory: {tool.persist_directory}")
    print(f"Collection: {tool.collection_name}")
    print(f"Matched rows: {loaded['stats']['matched_rows']}")
    print(f"Embedded (added): {summary['added']}")
    print(f"Skipped duplicates: {summary['skipped']}")
    print(f"Collection count: {tool.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
