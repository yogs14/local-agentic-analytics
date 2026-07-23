"""Build the retrieval gold TEMPLATE for the finance_news collection.

Generates ~50 candidate queries (Indonesian) from the EXISTING documents in
the ``finance_news`` ChromaDB collection, with ``relevant_doc_ids`` left EMPTY
for manual labeling. Nothing here decides relevance — that is the human's job.

Outputs:
- references/rag_gold/finance_news_retrieval_gold.json  (the template)
- references/rag_gold/finance_news_documents.csv        (doc_id -> headline
  lookup table to make manual labeling practical)

The script REFUSES to overwrite an existing gold file (it may already contain
manual labels) unless --force is given.

Example:
    python scripts/build_rag_gold_template.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.rag_eval import DEFAULT_GOLD_PATH

# Reuse the collection builder from the eval runner so both scripts always
# target the same ChromaDB configuration.
from run_rag_eval import DEFAULT_COLLECTION_NAME, build_chroma_tool


DOCUMENTS_CSV_PATH = (
    PROJECT_ROOT / "references" / "rag_gold" / "finance_news_documents.csv"
)

MONTH_NAMES_ID = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei",
    6: "Juni", 7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober",
    11: "November", 12: "Desember",
}

# Thematic queries: common finance-news topics, phrased in Indonesian. The
# {ticker} slot is filled from the collection's actual tickers.
THEMATIC_TEMPLATES = (
    "Berita tentang laporan pendapatan (earnings) {ticker}",
    "Berita tentang target harga analis untuk {ticker}",
    "Berita tentang kenaikan harga saham {ticker}",
    "Berita tentang penurunan harga saham {ticker}",
    "Berita tentang produk atau layanan baru {ticker}",
)

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with",
    "at", "by", "from", "as", "is", "are", "was", "be", "this", "that",
    "its", "it", "will", "after", "before", "into", "up", "down", "vs",
    "amid", "over", "under", "about", "than", "more", "less", "new",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the finance_news retrieval gold TEMPLATE (~50 queries, "
            "relevant_doc_ids empty for manual labeling)."
        )
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="ChromaDB collection to sample from (default: finance_news).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="Gold template JSON output path.",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=50,
        help="Target number of candidate queries (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for deterministic sampling (default: 42).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing gold file (may destroy manual labels!).",
    )
    return parser.parse_args()


def fetch_all_documents(chroma_tool) -> list[dict[str, str]]:
    """Read every document in the collection (id, headline, metadata)."""
    raw = chroma_tool.collection.get(include=["documents", "metadatas"])
    documents: list[dict[str, str]] = []
    for doc_id, document, metadata in zip(
        raw.get("ids", []), raw.get("documents", []), raw.get("metadatas", [])
    ):
        metadata = metadata or {}
        documents.append(
            {
                "doc_id": str(doc_id),
                "headline": str(document or ""),
                "ticker": str(metadata.get("ticker", "")),
                "date": str(metadata.get("date", "")),
                "publisher": str(metadata.get("publisher", "")),
            }
        )
    return documents


def _headline_keywords(headline: str, max_words: int = 5) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z'&.-]+", headline)
    keywords = [
        word for word in words
        if word.lower() not in _STOPWORDS and len(word) > 2
    ]
    return " ".join(keywords[:max_words])


def build_queries(
    documents: list[dict[str, str]],
    n_queries: int,
    seed: int,
) -> list[dict[str, object]]:
    """Assemble ticker-month, headline-seeded, and thematic candidate queries."""
    rng = random.Random(seed)
    tickers = sorted({doc["ticker"] for doc in documents if doc["ticker"]})
    by_ticker: dict[str, list[dict[str, str]]] = defaultdict(list)
    months_by_ticker: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for doc in documents:
        if not doc["ticker"]:
            continue
        by_ticker[doc["ticker"]].append(doc)
        match = re.match(r"(\d{4})-(\d{2})", doc["date"])
        if match:
            months_by_ticker[doc["ticker"]].add(
                (int(match.group(1)), int(match.group(2)))
            )

    queries: list[dict[str, object]] = []

    def add(query: str, query_type: str, hints: dict[str, str]) -> None:
        queries.append(
            {
                "query_id": f"R{len(queries) + 1:03d}",
                "query": query,
                "query_type": query_type,
                "labeling_hints": hints,
                "relevant_doc_ids": [],
            }
        )

    # 1. ticker + month browsing queries (~16, 4 per ticker).
    for ticker in tickers:
        months = sorted(months_by_ticker.get(ticker, set()))
        rng.shuffle(months)
        for year, month in months[:4]:
            add(
                f"Berita apa saja tentang saham {ticker} pada "
                f"{MONTH_NAMES_ID[month]} {year}?",
                "ticker_month",
                {"ticker": ticker, "month": f"{year}-{month:02d}"},
            )

    # 2. thematic queries (~20, 5 templates x 4 tickers).
    for template in THEMATIC_TEMPLATES:
        for ticker in tickers:
            add(
                template.format(ticker=ticker),
                "thematic",
                {"ticker": ticker},
            )

    # 3. headline-seeded keyword queries (fill up to n_queries).
    seeded_needed = max(n_queries - len(queries), 0)
    per_ticker = max(seeded_needed // max(len(tickers), 1), 1)
    for ticker in tickers:
        pool = list(by_ticker[ticker])
        rng.shuffle(pool)
        taken = 0
        for doc in pool:
            if taken >= per_ticker or len(queries) >= n_queries:
                break
            keywords = _headline_keywords(doc["headline"])
            if not keywords:
                continue
            add(
                f"Berita {ticker} terkait {keywords}",
                "headline_seeded",
                {
                    "ticker": ticker,
                    "seed_doc_id": doc["doc_id"],
                    "seed_headline": doc["headline"],
                },
            )
            taken += 1

    return queries[: max(n_queries, len(queries))]


def write_documents_csv(documents: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=("doc_id", "ticker", "date", "publisher", "headline")
        )
        writer.writeheader()
        for doc in sorted(documents, key=lambda d: (d["ticker"], d["date"])):
            writer.writerow(doc)


def main() -> int:
    args = parse_args()

    if args.output_path.is_file() and not args.force:
        print(
            f"Error: {args.output_path} already exists. It may contain manual "
            "labels; pass --force only if you intend to overwrite them."
        )
        return 1

    chroma_tool = build_chroma_tool(args.collection_name)
    if chroma_tool.count() == 0:
        print(
            f"Error: collection '{args.collection_name}' is empty. Jalankan "
            "python scripts/ingest_finance_news.py terlebih dahulu."
        )
        return 1

    documents = fetch_all_documents(chroma_tool)
    queries = build_queries(documents, args.n_queries, args.seed)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(queries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_documents_csv(documents, DOCUMENTS_CSV_PATH)

    type_counts: dict[str, int] = {}
    for query in queries:
        type_counts[str(query["query_type"])] = (
            type_counts.get(str(query["query_type"]), 0) + 1
        )

    print(f"Wrote {len(queries)} candidate queries -> {args.output_path}")
    print(f"Query types: {type_counts}")
    print(f"Document lookup table ({len(documents)} docs) -> {DOCUMENTS_CSV_PATH}")
    print(
        "\nLangkah manual berikutnya: isi relevant_doc_ids per query (pakai "
        "finance_news_documents.csv untuk mencari doc_id), lalu jalankan "
        "python scripts/run_rag_eval.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
