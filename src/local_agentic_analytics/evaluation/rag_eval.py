"""Retrieval evaluation metrics for the finance_news RAG path.

Computes standard ranking metrics against a manually labeled gold set:

- ``Hit@k``    - 1 when at least one relevant document appears in the top-k.
- ``Recall@k`` - fraction of ALL relevant documents found in the top-k.
- ``MRR``      - reciprocal rank of the first relevant document (0 when none
                 is retrieved at all).

The gold set (``references/rag_gold/finance_news_retrieval_gold.json``) is
labeled by a human: queries whose ``relevant_doc_ids`` list is still empty are
treated as UNLABELED and skipped (counted separately) — they never silently
count as zero.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Sequence

from local_agentic_analytics.core.config import PROJECT_ROOT


DEFAULT_GOLD_PATH = (
    PROJECT_ROOT / "references" / "rag_gold" / "finance_news_retrieval_gold.json"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "experiments" / "rag_eval.csv"
DEFAULT_KS = (1, 3, 5)

RAG_EVAL_COLUMNS = (
    "query_id",
    "query",
    "n_relevant",
    "n_retrieved",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "mrr",
    "first_relevant_rank",
    "retrieved_ids",
)

# retriever(query, top_k) -> ranked list of document ids (best first)
Retriever = Callable[[str, int], list[str]]


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """1.0 when any relevant id appears in the first ``k`` results."""
    if k < 1:
        raise ValueError("k must be positive")
    return 1.0 if any(doc_id in relevant_ids for doc_id in ranked_ids[:k]) else 0.0


def recall_at_k(
    ranked_ids: Sequence[str], relevant_ids: set[str], k: int
) -> float | None:
    """Fraction of relevant ids found in the first ``k`` results.

    ``None`` when there are no relevant ids (undefined, not zero).
    """
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant_ids:
        return None
    found = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant_ids)
    return found / len(relevant_ids)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """1/rank of the first relevant document; 0.0 when none is retrieved."""
    for index, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / index
    return 0.0


def first_relevant_rank(
    ranked_ids: Sequence[str], relevant_ids: set[str]
) -> int | None:
    for index, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return index
    return None


def load_rag_gold(path: str | Path = DEFAULT_GOLD_PATH) -> list[dict[str, Any]]:
    """Load and validate the retrieval gold JSON."""
    gold_path = Path(path)
    if not gold_path.is_file():
        raise FileNotFoundError(f"RAG gold file not found: {gold_path}")
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("RAG gold JSON must contain a list")
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"RAG gold item #{index} must be a dictionary")
        for field_name in ("query_id", "query"):
            value = item.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"RAG gold item #{index} must contain non-empty "
                    f"'{field_name}'"
                )
        if not isinstance(item.get("relevant_doc_ids", []), list):
            raise ValueError(
                f"RAG gold item #{index} 'relevant_doc_ids' must be a list"
            )
    return data


def evaluate_query(
    query_id: str,
    query: str,
    relevant_ids: set[str],
    ranked_ids: list[str],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, Any]:
    """Metrics for one labeled query against one ranked retrieval."""
    row: dict[str, Any] = {
        "query_id": query_id,
        "query": query,
        "n_relevant": len(relevant_ids),
        "n_retrieved": len(ranked_ids),
        "mrr": reciprocal_rank(ranked_ids, relevant_ids),
        "first_relevant_rank": first_relevant_rank(ranked_ids, relevant_ids),
        "retrieved_ids": "|".join(ranked_ids),
    }
    for k in ks:
        row[f"hit_at_{k}"] = hit_at_k(ranked_ids, relevant_ids, k)
        row[f"recall_at_{k}"] = recall_at_k(ranked_ids, relevant_ids, k)
    return row


def run_rag_evaluation(
    gold_items: list[dict[str, Any]],
    retriever: Retriever,
    ks: Sequence[int] = DEFAULT_KS,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Evaluate every LABELED gold query; return (rows, skipped_query_ids)."""
    top_k = max(ks)
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for item in gold_items:
        query_id = str(item["query_id"])
        relevant_ids = {
            str(doc_id) for doc_id in item.get("relevant_doc_ids", []) if doc_id
        }
        if not relevant_ids:
            skipped.append(query_id)
            continue
        ranked_ids = [str(doc_id) for doc_id in retriever(str(item["query"]), top_k)]
        rows.append(
            evaluate_query(query_id, str(item["query"]), relevant_ids, ranked_ids, ks)
        )
    return rows, skipped


def summarize_rag_results(
    rows: list[dict[str, Any]],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, Any]:
    """Aggregate per-query rows into mean metrics."""
    summary: dict[str, Any] = {"n_queries": len(rows)}
    if not rows:
        return summary
    summary["mrr"] = statistics.fmean(float(row["mrr"]) for row in rows)
    for k in ks:
        summary[f"hit_at_{k}"] = statistics.fmean(
            float(row[f"hit_at_{k}"]) for row in rows
        )
        recalls = [
            float(row[f"recall_at_{k}"])
            for row in rows
            if row.get(f"recall_at_{k}") is not None
        ]
        summary[f"recall_at_{k}"] = (
            statistics.fmean(recalls) if recalls else None
        )
    return summary


def write_rag_results(
    rows: list[dict[str, Any]],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Write per-query metric rows to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(RAG_EVAL_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: _format_cell(row.get(column))
                    for column in RAG_EVAL_COLUMNS
                }
            )
    return path


def _format_cell(value: Any) -> Any:
    if value is None:
        return ""
    return value
