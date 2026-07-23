import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.rag_eval import (
    RAG_EVAL_COLUMNS,
    evaluate_query,
    first_relevant_rank,
    hit_at_k,
    load_rag_gold,
    reciprocal_rank,
    recall_at_k,
    run_rag_evaluation,
    summarize_rag_results,
    write_rag_results,
)


RANKED = ["d1", "d2", "d3", "d4", "d5"]


def test_hit_at_k():
    assert hit_at_k(RANKED, {"d3"}, 3) == 1.0
    assert hit_at_k(RANKED, {"d3"}, 2) == 0.0
    assert hit_at_k(RANKED, {"nope"}, 5) == 0.0
    with pytest.raises(ValueError):
        hit_at_k(RANKED, {"d1"}, 0)


def test_recall_at_k():
    assert recall_at_k(RANKED, {"d1", "d4"}, 1) == pytest.approx(0.5)
    assert recall_at_k(RANKED, {"d1", "d4"}, 5) == pytest.approx(1.0)
    assert recall_at_k(RANKED, {"d1", "zz"}, 5) == pytest.approx(0.5)
    assert recall_at_k(RANKED, set(), 5) is None


def test_reciprocal_rank_and_first_rank():
    assert reciprocal_rank(RANKED, {"d1"}) == 1.0
    assert reciprocal_rank(RANKED, {"d3"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"nope"}) == 0.0
    assert first_relevant_rank(RANKED, {"d3"}) == 3
    assert first_relevant_rank(RANKED, {"nope"}) is None


def test_evaluate_query_builds_full_row():
    row = evaluate_query("R001", "q", {"d2"}, RANKED)
    assert row["hit_at_1"] == 0.0
    assert row["hit_at_3"] == 1.0
    assert row["recall_at_5"] == 1.0
    assert row["mrr"] == pytest.approx(0.5)
    assert row["first_relevant_rank"] == 2
    assert row["retrieved_ids"] == "|".join(RANKED)


def test_run_rag_evaluation_skips_unlabeled():
    gold = [
        {"query_id": "R001", "query": "a", "relevant_doc_ids": ["d1"]},
        {"query_id": "R002", "query": "b", "relevant_doc_ids": []},
    ]

    def retriever(query, top_k):
        assert top_k == 5
        return RANKED

    rows, skipped = run_rag_evaluation(gold, retriever)

    assert len(rows) == 1
    assert rows[0]["query_id"] == "R001"
    assert skipped == ["R002"]


def test_summarize_rag_results():
    gold = [
        {"query_id": "R001", "query": "a", "relevant_doc_ids": ["d1"]},
        {"query_id": "R002", "query": "b", "relevant_doc_ids": ["d5"]},
    ]
    rows, _ = run_rag_evaluation(gold, lambda q, k: RANKED)

    summary = summarize_rag_results(rows)

    assert summary["n_queries"] == 2
    assert summary["hit_at_1"] == pytest.approx(0.5)
    assert summary["hit_at_5"] == pytest.approx(1.0)
    assert summary["mrr"] == pytest.approx((1.0 + 0.2) / 2)

    assert summarize_rag_results([]) == {"n_queries": 0}


def test_load_rag_gold_validates(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps([{"query_id": "R1", "query": "x", "relevant_doc_ids": []}]),
        encoding="utf-8",
    )
    assert load_rag_gold(path)[0]["query_id"] == "R1"

    path.write_text(json.dumps([{"query": "no id"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_rag_gold(path)

    with pytest.raises(FileNotFoundError):
        load_rag_gold(tmp_path / "missing.json")


def test_write_rag_results(tmp_path):
    gold = [{"query_id": "R001", "query": "a", "relevant_doc_ids": ["d1"]}]
    rows, _ = run_rag_evaluation(gold, lambda q, k: RANKED)

    path = write_rag_results(rows, tmp_path / "rag_eval.csv")

    content = path.read_text(encoding="utf-8").splitlines()
    assert content[0] == ",".join(RAG_EVAL_COLUMNS)
    assert "R001" in content[1]
