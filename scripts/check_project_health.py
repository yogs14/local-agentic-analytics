from pathlib import Path
import os
import sys

import duckdb
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core.config import load_config


def main() -> int:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)

    db_path = _get_duckdb_path()
    finance_news_csv = PROJECT_ROOT / "data" / "raw" / "finance" / "raw_analyst_ratings.csv"
    checks = [
        ("env_file", ".env tersedia", env_path.exists()),
        ("ollama_base_url", "OLLAMA_BASE_URL tersedia", bool(os.getenv("OLLAMA_BASE_URL"))),
        ("ollama_model", "OLLAMA_MODEL tersedia", bool(os.getenv("OLLAMA_MODEL"))),
        ("duckdb_database", "DuckDB database tersedia", db_path.exists()),
        (
            "electric_power_table",
            "Tabel electric_power tersedia",
            _table_exists(db_path, "electric_power") if db_path.exists() else False,
        ),
        (
            "stock_prices_table",
            "Tabel stock_prices (finance) tersedia",
            _table_exists(db_path, "stock_prices") if db_path.exists() else False,
        ),
        (
            "finance_news_csv",
            "data/raw/finance/raw_analyst_ratings.csv tersedia",
            finance_news_csv.exists(),
        ),
        (
            "finance_news_collection",
            "Koleksi ChromaDB finance_news berisi dokumen",
            _chromadb_collection_has_documents("finance_news"),
        ),
        (
            "reports_figures",
            "reports/figures/ tersedia",
            (PROJECT_ROOT / "reports" / "figures").exists(),
        ),
        (
            "reports_latex",
            "reports/latex/ tersedia",
            (PROJECT_ROOT / "reports" / "latex").exists(),
        ),
        (
            "reports_pdf",
            "reports/pdf/ tersedia",
            (PROJECT_ROOT / "reports" / "pdf").exists(),
        ),
    ]

    print("Project health check:")
    for _, message, ok in checks:
        status = "OK" if ok else "FAIL"
        print(f"- [{status}] {message}")

    failed = [name for name, _, ok in checks if not ok]
    if failed:
        print()
        print("Health check gagal. Perbaiki item FAIL sebelum demo atau eksperimen.")
        if "env_file" in failed:
            print("- Buat .env dari .env.example: Copy-Item .env.example .env")
        if "duckdb_database" in failed or "electric_power_table" in failed:
            print("- Jalankan ingestion: python scripts/ingest_energy.py")
        if "stock_prices_table" in failed:
            print("- Jalankan ingestion finance: python scripts/ingest_finance_prices.py")
        if "finance_news_collection" in failed:
            print("- Jalankan ingestion berita: python scripts/ingest_finance_news.py")
        if "finance_news_csv" in failed:
            print(
                "- Letakkan raw_analyst_ratings.csv di data/raw/finance/ "
                "sebelum ingestion berita."
            )
        return 1

    print()
    print("Health check sukses. Project siap untuk eksperimen dasar.")
    return 0


def _get_duckdb_path() -> Path:
    config = load_config("duckdb.yaml")
    configured_path = config.get("duckdb", {}).get("database_path")
    if not isinstance(configured_path, str) or not configured_path.strip():
        return PROJECT_ROOT / "databases" / "duckdb" / "analytics.duckdb"

    path = Path(configured_path)
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def _chromadb_collection_has_documents(collection_name: str) -> bool:
    try:
        from local_agentic_analytics.tools.chromadb_tool import (
            DEFAULT_EMBEDDING_MODEL,
            DEFAULT_PERSIST_DIRECTORY,
            ChromaDBTool,
        )

        config = load_config("chromadb.yaml")
        chroma_config = config.get("chromadb", {}) if isinstance(config, dict) else {}
        embedding_config = (
            config.get("embedding", {}) if isinstance(config, dict) else {}
        )
        persist_directory = Path(
            str(chroma_config.get("persist_directory", DEFAULT_PERSIST_DIRECTORY))
        )
        if not persist_directory.is_absolute():
            persist_directory = PROJECT_ROOT / persist_directory
        if not persist_directory.exists():
            return False

        tool = ChromaDBTool(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_model_name=str(
                embedding_config.get("model", DEFAULT_EMBEDDING_MODEL)
            ),
        )
        return tool.count() > 0
    except Exception:
        return False


def _table_exists(db_path: Path, table_name: str) -> bool:
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            result = con.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = ?
                """,
                [table_name],
            ).fetchone()
        finally:
            con.close()
    except Exception:
        return False

    return bool(result and result[0] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
