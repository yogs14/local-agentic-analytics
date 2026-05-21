"""Build a small dummy ChromaDB collection for initial RAG experiments."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.tools.chromadb_tool import ChromaDBTool


DUMMY_DOCUMENTS = [
    "Dataset electric_power berisi konsumsi daya listrik rumah tangga per menit.",
    "Kolom Global_active_power menyimpan daya aktif global dalam satuan kilowatt.",
    "Kolom datetime dapat digunakan untuk filter tanggal, bulan, atau jam.",
]
DUMMY_METADATAS = [
    {"source": "dummy", "topic": "dataset"},
    {"source": "dummy", "topic": "column"},
    {"source": "dummy", "topic": "time_filter"},
]


def main() -> int:
    tool = ChromaDBTool.from_config()
    existing_count = tool.count()

    if existing_count > 0:
        print(
            f"Collection '{tool.collection_name}' already has {existing_count} "
            "documents. Skipping rebuild."
        )
        return 0

    summary = tool.add_documents(DUMMY_DOCUMENTS, DUMMY_METADATAS)
    print(f"Persist directory: {tool.persist_directory}")
    print(f"Collection: {tool.collection_name}")
    print(f"Added documents: {summary['added']}")
    print(f"Skipped documents: {summary['skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
