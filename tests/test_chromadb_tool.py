from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.tools.chromadb_tool import ChromaDBTool


class FakeEmbeddingFunction:
    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input:
            embeddings.append(
                [
                    float(len(text)),
                    float(text.lower().count("daya")),
                    float(text.lower().count("tanggal")),
                ]
            )
        return embeddings


def test_add_documents_persists_and_skips_duplicates(tmp_path):
    tool = ChromaDBTool(
        persist_directory=tmp_path / "chromadb",
        collection_name="test_collection",
        embedding_function=FakeEmbeddingFunction(),
        embedding_model_name="fake",
    )
    documents = [
        "Kolom Global_active_power menyimpan daya aktif.",
        "Kolom datetime dipakai untuk filter tanggal.",
    ]
    metadatas = [
        {"source": "test", "topic": "power"},
        {"source": "test", "topic": "datetime"},
    ]

    first_summary = tool.add_documents(documents, metadatas)
    second_summary = tool.add_documents(documents, metadatas)

    assert first_summary["added"] == 2
    assert first_summary["skipped"] == 0
    assert second_summary["added"] == 0
    assert second_summary["skipped"] == 2
    assert tool.count() == 2

    reopened_tool = ChromaDBTool(
        persist_directory=tmp_path / "chromadb",
        collection_name="test_collection",
        embedding_function=FakeEmbeddingFunction(),
        embedding_model_name="fake",
    )

    assert reopened_tool.count() == 2


def test_query_returns_ranked_matches(tmp_path):
    tool = ChromaDBTool(
        persist_directory=tmp_path / "chromadb",
        collection_name="test_collection",
        embedding_function=FakeEmbeddingFunction(),
        embedding_model_name="fake",
    )
    tool.add_documents(
        documents=[
            "Kolom Global_active_power menyimpan daya aktif.",
            "Kolom datetime dipakai untuk filter tanggal.",
        ],
        metadatas=[
            {"topic": "power"},
            {"topic": "datetime"},
        ],
    )

    matches = tool.query("daya aktif", top_k=1)

    assert len(matches) == 1
    assert matches[0]["document"]
    assert matches[0]["metadata"]["topic"] in {"power", "datetime"}
    assert isinstance(matches[0]["distance"], float)


def test_query_empty_collection_returns_empty_list(tmp_path):
    tool = ChromaDBTool(
        persist_directory=tmp_path / "chromadb",
        collection_name="test_collection",
        embedding_function=FakeEmbeddingFunction(),
        embedding_model_name="fake",
    )

    assert tool.query("daya aktif") == []


def test_add_documents_validates_lengths(tmp_path):
    tool = ChromaDBTool(
        persist_directory=tmp_path / "chromadb",
        collection_name="test_collection",
        embedding_function=FakeEmbeddingFunction(),
        embedding_model_name="fake",
    )

    with pytest.raises(ValueError, match="same length"):
        tool.add_documents(["satu dokumen"], [])
