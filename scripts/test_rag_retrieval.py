"""Smoke test for ChromaDB RAG document retrieval."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.tools.chromadb_tool import ChromaDBTool


QUERY_TEXT = "Apa satuan Global_active_power?"
TOP_K = 3


def main() -> int:
    tool = ChromaDBTool.from_config()
    results = tool.query(QUERY_TEXT, top_k=TOP_K)

    if not results:
        print(
            "Tidak ada dokumen ditemukan. Jalankan python scripts/build_chromadb.py terlebih dahulu."
        )
        return 1

    print(f"Query: {QUERY_TEXT}")
    print(f"Top K: {TOP_K}")
    print()

    for rank, result in enumerate(results, start=1):
        score = result.get("score")
        distance = result.get("distance")
        score_text = score if score is not None else distance

        print(f"Rank: {rank}")
        print(f"Score: {score_text if score_text is not None else '-'}")
        print(f"Document: {result.get('document', '')}")
        print(f"Metadata: {result.get('metadata', {})}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
