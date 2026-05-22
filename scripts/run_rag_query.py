"""Run a simple ChromaDB + Ollama RAG query."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.tools.chromadb_tool import ChromaDBTool
from local_agentic_analytics.tools.ollama_tool import OllamaTool


TOP_K = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple RAG query using ChromaDB retrieval and local Ollama."
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Question to answer using retrieved ChromaDB context.",
    )
    return parser.parse_args()


def build_rag_prompt(query: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[{index}] {item.get('document', '')}"
        for index, item in enumerate(contexts, start=1)
    )
    return f"""Anda adalah asisten RAG lokal.

Jawab pertanyaan user dalam bahasa Indonesia berdasarkan konteks berikut.
Gunakan hanya informasi dari konteks.
Jika konteks tidak cukup, katakan bahwa informasi pada konteks belum cukup.
Jawab singkat dan jelas.

Konteks:
{context_text}

Pertanyaan:
{query}

Jawaban:
"""


def print_retrieved_context(contexts: list[dict]) -> None:
    print("retrieved_context:")
    for rank, context in enumerate(contexts, start=1):
        score = context.get("score")
        distance = context.get("distance")
        score_text = score if score is not None else distance

        print(f"- rank: {rank}")
        print(f"  score: {score_text if score_text is not None else '-'}")
        print(f"  document: {context.get('document', '')}")
        print(f"  metadata: {context.get('metadata', {})}")


def main() -> int:
    args = parse_args()
    query = " ".join(args.query).strip()
    latency: dict[str, float] = {}
    total_start = perf_counter()

    chromadb_tool = ChromaDBTool.from_config()

    retrieval_start = perf_counter()
    contexts = chromadb_tool.query(query, top_k=TOP_K)
    latency["retrieval"] = perf_counter() - retrieval_start

    if not contexts:
        latency["total"] = perf_counter() - total_start
        print(f"query: {query}")
        print(
            "Tidak ada dokumen ditemukan. Jalankan python scripts/build_chromadb.py terlebih dahulu."
        )
        print("latency:")
        for step_name, seconds in latency.items():
            print(f"- {step_name}: {seconds:.3f}s")
        return 1

    prompt = build_rag_prompt(query, contexts)
    ollama_tool = OllamaTool.from_config()

    generation_start = perf_counter()
    final_answer = ollama_tool.generate(
        prompt=prompt,
        temperature=0.1,
        max_tokens=256,
    )
    latency["generation"] = perf_counter() - generation_start
    latency["total"] = perf_counter() - total_start

    print(f"query: {query}")
    print()
    print_retrieved_context(contexts)
    print()
    print("final_answer:")
    print(final_answer)
    print()
    print("latency:")
    for step_name, seconds in latency.items():
        print(f"- {step_name}: {seconds:.3f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
