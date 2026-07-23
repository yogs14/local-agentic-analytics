"""Evaluate finance_news retrieval against the manually labeled gold set.

Computes Hit@k, Recall@k (k=1,3,5) and MRR for every LABELED query in
``references/rag_gold/finance_news_retrieval_gold.json`` (queries whose
``relevant_doc_ids`` is still empty are skipped and reported). Results go to
``reports/experiments/rag_eval.csv``.

Examples:
    python scripts/run_rag_eval.py
    python scripts/run_rag_eval.py --top-k 10 --collection-name finance_news
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core.config import load_config
from local_agentic_analytics.evaluation.rag_eval import (
    DEFAULT_GOLD_PATH,
    DEFAULT_KS,
    DEFAULT_OUTPUT_PATH,
    load_rag_gold,
    run_rag_evaluation,
    summarize_rag_results,
    write_rag_results,
)
from local_agentic_analytics.tools.chromadb_tool import (
    DEFAULT_PERSIST_DIRECTORY,
    ChromaDBTool,
)


DEFAULT_COLLECTION_NAME = "finance_news"


def build_chroma_tool(collection_name: str) -> ChromaDBTool:
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
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate finance_news retrieval (Hit@k, Recall@k, MRR) against "
            "the labeled gold set."
        )
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="Path to the retrieval gold JSON.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Per-query metrics CSV output.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="ChromaDB collection to evaluate (default: finance_news).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=max(DEFAULT_KS),
        help="Retrieval depth (>= largest k; default: 5).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.top_k < max(DEFAULT_KS):
        print(f"Error: --top-k must be >= {max(DEFAULT_KS)}")
        return 1

    try:
        gold_items = load_rag_gold(args.gold_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        print(
            "Buat template gold dulu: python scripts/build_rag_gold_template.py"
        )
        return 1

    chroma_tool = build_chroma_tool(args.collection_name)
    if chroma_tool.count() == 0:
        print(
            f"Error: collection '{args.collection_name}' is empty. Jalankan "
            "python scripts/ingest_finance_news.py terlebih dahulu."
        )
        return 1

    def retriever(query: str, top_k: int) -> list[str]:
        matches = chroma_tool.query(text=query, top_k=top_k)
        return [str(match.get("id", "")) for match in matches]

    rows, skipped = run_rag_evaluation(gold_items, retriever)
    write_rag_results(rows, args.output_path)
    summary = summarize_rag_results(rows)

    print("RAG retrieval evaluation (finance_news):")
    print(f"- labeled queries evaluated : {summary.get('n_queries', 0)}")
    print(f"- unlabeled queries skipped : {len(skipped)}")
    if summary.get("n_queries"):
        print(f"- MRR       : {summary['mrr']:.4f}")
        for k in DEFAULT_KS:
            recall = summary.get(f"recall_at_{k}")
            recall_text = f"{recall:.4f}" if recall is not None else "-"
            print(
                f"- Hit@{k}: {summary[f'hit_at_{k}']:.4f}   "
                f"Recall@{k}: {recall_text}"
            )
    else:
        print(
            "Tidak ada query berlabel. Isi relevant_doc_ids di "
            f"{args.gold_path} lalu jalankan ulang."
        )
    print(f"- output    : {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
