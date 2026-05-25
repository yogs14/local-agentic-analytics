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
