"""Export RAG answers into a manual rating sheet for two raters.

For every query in the retrieval gold file, runs the SAME finance_news RAG
path used by ``scripts/run_finance_rag_query.py`` (retrieval + ReporterAgent
narrative) and writes a CSV with EMPTY score columns for two raters, to be
filled according to ``docs/rag_generation_rubric.md`` (faithfulness & answer
relevance, 1-5).

The script refuses to overwrite an existing sheet (it may contain manual
ratings) unless --force is given.

Example:
    python scripts/export_rag_answers_for_rating.py --limit 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.reporter_agent import ReporterAgent
from local_agentic_analytics.evaluation.rag_eval import (
    DEFAULT_GOLD_PATH,
    load_rag_gold,
)

from run_finance_rag_query import run_rag_query
from run_rag_eval import DEFAULT_COLLECTION_NAME, build_chroma_tool


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "rag_rating_sheet.csv"
)

RATING_COLUMNS = (
    "query_id",
    "query",
    "retrieved_headlines",
    "answer",
    "rater1_faithfulness",
    "rater1_relevance",
    "notes_rater1",
    "rater2_faithfulness",
    "rater2_relevance",
    "notes_rater2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate answers for every RAG gold query and export a manual "
            "rating sheet (two raters, faithfulness & relevance 1-5)."
        )
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="Retrieval gold JSON with the queries to answer.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Rating sheet CSV output.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="ChromaDB collection (default: finance_news).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Headlines retrieved per query (default: 5).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max queries to answer (smoke tests).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing sheet (may destroy manual ratings!).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.output_path.is_file() and not args.force:
        print(
            f"Error: {args.output_path} already exists. It may contain manual "
            "ratings; pass --force only if you intend to overwrite them."
        )
        return 1
    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be greater than 0")
        return 1

    try:
        gold_items = load_rag_gold(args.gold_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        print(
            "Buat template gold dulu: python scripts/build_rag_gold_template.py"
        )
        return 1
    if args.limit is not None:
        gold_items = gold_items[: args.limit]

    chroma_tool = build_chroma_tool(args.collection_name)
    if chroma_tool.count() == 0:
        print(
            f"Error: collection '{args.collection_name}' is empty. Jalankan "
            "python scripts/ingest_finance_news.py terlebih dahulu."
        )
        return 1
    reporter = ReporterAgent()

    rows: list[dict[str, str]] = []
    for index, item in enumerate(gold_items, start=1):
        query = str(item["query"])
        print(f"[{index}/{len(gold_items)}] {query}")
        result = run_rag_query(
            query,
            chroma_tool=chroma_tool,
            reporter=reporter,
            top_k=args.top_k,
        )
        headlines = " || ".join(
            f"[{headline.get('ticker', '')} {headline.get('date', '')}] "
            f"{headline.get('headline', '')}"
            for headline in result.get("retrieved_headlines", [])
        )
        rows.append(
            {
                "query_id": str(item["query_id"]),
                "query": query,
                "retrieved_headlines": headlines,
                "answer": str(result.get("answer", "")),
            }
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(RATING_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: row.get(column, "") for column in RATING_COLUMNS}
            )

    print(f"\nWrote {len(rows)} items -> {args.output_path}")
    print(
        "Dua penilai mengisi kolom rater1_*/rater2_* secara independen sesuai "
        "docs/rag_generation_rubric.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
