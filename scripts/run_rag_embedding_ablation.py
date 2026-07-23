"""Embedding-model ablation for finance_news retrieval (Fase 4.2).

Re-embeds the SAME documents into one side collection per candidate embedding
model, then evaluates every candidate against the SAME labeled gold set
(Hit@k, Recall@k, MRR). Document ids are deterministic on (document,
metadata) — not on the embedding — so gold labels stay valid across
collections.

Default candidates:
- sentence-transformers/all-MiniLM-L6-v2            (current default, English)
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  (multilingual; hypothesis: better for Indonesian queries over English
  headlines)

Outputs:
- reports/experiments/rag_eval_<slug>.csv            (per-query rows per model)
- reports/experiments/rag_embedding_ablation.csv     (comparison table)

Example:
    python scripts/run_rag_embedding_ablation.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core.config import load_config
from local_agentic_analytics.evaluation.rag_eval import (
    DEFAULT_GOLD_PATH,
    DEFAULT_KS,
    load_rag_gold,
    run_rag_evaluation,
    summarize_rag_results,
    write_rag_results,
)
from local_agentic_analytics.tools.chromadb_tool import (
    DEFAULT_PERSIST_DIRECTORY,
    ChromaDBTool,
)

from run_rag_eval import DEFAULT_COLLECTION_NAME, build_chroma_tool


DEFAULT_EMBEDDING_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
ABLATION_SUMMARY_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "rag_embedding_ablation.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare embedding models on the same finance_news documents and "
            "labeled retrieval gold set."
        )
    )
    parser.add_argument(
        "--embedding-models",
        default=",".join(DEFAULT_EMBEDDING_MODELS),
        help="Comma-separated sentence-transformers model names.",
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="Labeled retrieval gold JSON.",
    )
    parser.add_argument(
        "--source-collection",
        default=DEFAULT_COLLECTION_NAME,
        help="Collection holding the documents (default: finance_news).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=max(DEFAULT_KS),
        help="Retrieval depth (default: 5).",
    )
    return parser.parse_args()


def _slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")[-40:]


def _persist_path() -> Path:
    config = load_config("chromadb.yaml")
    chroma_config = config.get("chromadb", {}) if isinstance(config, dict) else {}
    persist_directory = chroma_config.get(
        "persist_directory", str(DEFAULT_PERSIST_DIRECTORY)
    )
    path = Path(persist_directory)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def fetch_documents(source_tool: ChromaDBTool) -> tuple[list[str], list[dict]]:
    raw = source_tool.collection.get(include=["documents", "metadatas"])
    documents = [str(document) for document in raw.get("documents", [])]
    metadatas = [dict(metadata or {}) for metadata in raw.get("metadatas", [])]
    return documents, metadatas


def build_ablation_collection(
    model_name: str,
    documents: list[str],
    metadatas: list[dict],
) -> ChromaDBTool:
    """(Re)embed the documents with ``model_name`` into a side collection."""
    tool = ChromaDBTool(
        persist_directory=_persist_path(),
        collection_name=f"finance_news_abl_{_slug(model_name)}",
        embedding_model_name=model_name,
    )
    if tool.count() < len(documents):
        summary = tool.add_documents(documents, metadatas)
        print(
            f"  embedded {summary['added']} docs "
            f"(skipped {summary['skipped']} already present)"
        )
    return tool


def main() -> int:
    args = parse_args()
    model_names = [
        name.strip()
        for name in args.embedding_models.split(",")
        if name.strip()
    ]
    if not model_names:
        print("Error: no embedding models given")
        return 1

    try:
        gold_items = load_rag_gold(args.gold_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    source_tool = build_chroma_tool(args.source_collection)
    if source_tool.count() == 0:
        print(f"Error: collection '{args.source_collection}' is empty.")
        return 1
    documents, metadatas = fetch_documents(source_tool)
    print(f"Source documents: {len(documents)} from '{args.source_collection}'")

    summary_rows: list[dict[str, object]] = []
    for model_name in model_names:
        print(f"\n=== {model_name} ===")
        tool = build_ablation_collection(model_name, documents, metadatas)

        def retriever(query: str, top_k: int) -> list[str]:
            matches = tool.query(text=query, top_k=top_k)
            return [str(match.get("id", "")) for match in matches]

        rows, skipped = run_rag_evaluation(gold_items, retriever)
        per_query_path = (
            PROJECT_ROOT
            / "reports"
            / "experiments"
            / f"rag_eval_{_slug(model_name)}.csv"
        )
        write_rag_results(rows, per_query_path)
        summary = summarize_rag_results(rows)
        summary_row: dict[str, object] = {
            "embedding_model": model_name,
            "n_queries": summary.get("n_queries", 0),
            "n_skipped_unlabeled": len(skipped),
            "mrr": summary.get("mrr"),
        }
        for k in DEFAULT_KS:
            summary_row[f"hit_at_{k}"] = summary.get(f"hit_at_{k}")
            summary_row[f"recall_at_{k}"] = summary.get(f"recall_at_{k}")
        summary_rows.append(summary_row)

        print(f"  n={summary.get('n_queries', 0)}  MRR={summary.get('mrr', 0):.4f}")
        for k in DEFAULT_KS:
            print(
                f"  Hit@{k}: {summary.get(f'hit_at_{k}', 0):.4f}   "
                f"Recall@{k}: {summary.get(f'recall_at_{k}', 0):.4f}"
            )
        print(f"  per-query rows: {per_query_path}")

    ABLATION_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary_rows[0].keys())
    with ABLATION_SUMMARY_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"\nComparison table: {ABLATION_SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
