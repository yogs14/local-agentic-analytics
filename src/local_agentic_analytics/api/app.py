"""FastAPI backend service untuk local-agentic-analytics."""
from __future__ import annotations

import asyncio
import re
import shutil
import statistics
import uuid
from pathlib import Path

import duckdb
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from local_agentic_analytics.api.models import (
    AskRequest,
    AskResponse,
    ColumnInfo,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    ReportRequest,
    ReportResponse,
    SessionInfo,
    ToolCallSummary,
    UploadResponse,
)
from local_agentic_analytics.api.session_store import SessionStore
from local_agentic_analytics.core.config import PROJECT_ROOT, load_config
from local_agentic_analytics.core.profile_generator import (
    generate_profile_from_duckdb,
    get_table_summary,
)
from local_agentic_analytics.graph.generic_report_workflow import (
    GenericReportWorkflow,
)
from local_agentic_analytics.graph.workflow import SequentialAnalyticsWorkflow
from local_agentic_analytics.tools.duckdb_tool import DuckDBTool
from local_agentic_analytics.tools.ollama_tool import OllamaTool


app = FastAPI(
    title="Local Agentic Analytics API",
    version="1.0.0",
    description="Backend service untuk Q&A dan laporan PDF dari CSV sembarang.",
)

# CORS - izinkan semua origin karena ini local dev tool. Frontend Next.js
# berjalan di port berbeda.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_store = SessionStore()

# Batas upload.
_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
_MAX_ROW_COUNT = 500_000


# -- Helpers ----------------------------------------------------------------

def _get_db_path() -> str:
    """Baca database_path dari configs/duckdb.yaml via load_config()."""
    config = load_config("duckdb.yaml")
    raw = config.get("duckdb", {}).get("database_path")
    if not isinstance(raw, str) or not raw.strip():
        return str(PROJECT_ROOT / "databases" / "duckdb" / "analytics.duckdb")
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _sanitize_table_name(raw: str) -> str:
    """Buat nama tabel DuckDB yang aman dari nama file (tanpa ekstensi)."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = "t_" + cleaned
    return cleaned[:60]


def _table_exists(duckdb_tool: DuckDBTool, table_name: str) -> bool:
    escaped = table_name.replace("'", "''")
    result = duckdb_tool.execute_query(
        "SELECT COUNT(*) AS n FROM information_schema.tables "
        f"WHERE table_name = '{escaped}'"
    )
    return int(result["n"].iloc[0]) > 0


def _make_unique_table_name(base_name: str, duckdb_tool: DuckDBTool) -> str:
    """Pastikan nama tabel unik di DuckDB."""
    if not _table_exists(duckdb_tool, base_name):
        return base_name
    for suffix in range(2, 100):
        candidate = f"{base_name}_{suffix}"
        if not _table_exists(duckdb_tool, candidate):
            return candidate
    raise ValueError("Tidak dapat membuat nama tabel unik.")


def _detect_delimiter(raw_bytes: bytes) -> str:
    """Deteksi delimiter dari 2048 bytes pertama file CSV."""
    sample = raw_bytes[:2048].decode("utf-8", errors="ignore")
    lines = [line for line in sample.splitlines() if line.strip()]
    if not lines:
        return ","

    candidates = {",": [], ";": [], "\t": []}
    for line in lines:
        for delimiter in candidates:
            candidates[delimiter].append(line.count(delimiter))

    best_delimiter = ","
    best_score = 0.0
    for delimiter, counts in candidates.items():
        median = statistics.median(counts) if counts else 0
        if median > best_score:
            best_score = median
            best_delimiter = delimiter
    return best_delimiter


def _error_detail(message: str) -> dict:
    return {"success": False, "error": message}


def _default_model_tag() -> str:
    """Model tag the server falls back to when a request omits ``model``."""
    return OllamaTool.from_config().model


def _resolve_model_tag(requested: str | None) -> str:
    if requested and requested.strip():
        return requested.strip()
    return _default_model_tag()


def _cleanup_upload(
    duckdb_tool: DuckDBTool | None,
    table_name: str | None,
    output_dir: Path,
) -> None:
    if duckdb_tool is not None and table_name:
        try:
            duckdb_tool.execute_query(f'DROP TABLE IF EXISTS "{table_name}"')
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
    shutil.rmtree(output_dir, ignore_errors=True)


# -- Exception handlers -----------------------------------------------------

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return await http_exception_handler(request, exc)


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=_error_detail("Terjadi kesalahan internal pada server."),
    )


# -- Endpoints --------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    async def _check_ollama() -> bool:
        try:
            return await asyncio.to_thread(
                lambda: OllamaTool.from_config().check_connection()
            )
        except Exception:  # noqa: BLE001
            return False

    async def _check_duckdb() -> bool:
        def _probe() -> bool:
            tool = DuckDBTool(_get_db_path())
            try:
                tool.connect()
                tool.execute_query("SELECT 1")
                return True
            finally:
                tool.close()

        try:
            return await asyncio.to_thread(_probe)
        except Exception:  # noqa: BLE001
            return False

    ollama_ok, duckdb_ok = await asyncio.gather(
        _check_ollama(), _check_duckdb()
    )
    status = "ok" if ollama_ok and duckdb_ok else "degraded"
    return HealthResponse(
        status=status, ollama_ok=ollama_ok, duckdb_ok=duckdb_ok
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """Model Ollama yang terdaftar di ``configs/models/*.yaml``.

    ``installed`` dicek langsung ke Ollama (``/api/tags``) supaya frontend
    tidak menawarkan model yang belum di-pull. Bila Ollama tidak terjangkau,
    hanya model default (dari env ``OLLAMA_MODEL``) yang ditandai terpasang.
    """
    default_tag = await asyncio.to_thread(_default_model_tag)

    async def _installed_tags() -> set[str]:
        try:
            return await asyncio.to_thread(
                lambda: OllamaTool.from_config().list_installed_tags()
            )
        except Exception:  # noqa: BLE001
            return set()

    installed = await _installed_tags()

    models: list[ModelInfo] = []
    models_dir = PROJECT_ROOT / "configs" / "models"
    for path in sorted(models_dir.glob("*.yaml")):
        try:
            raw = load_config(f"models/{path.name}")
        except Exception:  # noqa: BLE001
            continue
        entry = raw.get("model", {})
        if not isinstance(entry, dict):
            continue
        tag = str(entry.get("ollama_tag", "")).strip()
        if not tag:
            continue
        models.append(
            ModelInfo(
                key=str(entry.get("key", path.stem)),
                ollama_tag=tag,
                param_count=str(entry.get("param_count", "")),
                quantization=str(entry.get("quantization", "")),
                is_default=(tag == default_tag),
                installed=(tag in installed) if installed else (tag == default_tag),
            )
        )
    return ModelsResponse(models=models, default_model=default_tag)


@app.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)) -> UploadResponse:
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=_error_detail("File harus berekstensi .csv"),
        )

    content = await file.read()
    if len(content) > _MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=_error_detail("Ukuran file melebihi batas 50 MB."),
        )
    if not content.strip():
        raise HTTPException(
            status_code=400, detail=_error_detail("File CSV kosong.")
        )

    delimiter = _detect_delimiter(content)
    session_id = str(uuid.uuid4())
    output_dir = PROJECT_ROOT / "reports" / "sessions" / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / filename

    duckdb_tool: DuckDBTool | None = None
    table_name: str | None = None
    try:
        csv_path.write_bytes(content)
        db_path = _get_db_path()
        duckdb_tool = DuckDBTool(db_path)
        table_name = _make_unique_table_name(
            _sanitize_table_name(Path(filename).stem), duckdb_tool
        )
        csv_sql_path = str(csv_path).replace("\\", "/").replace("'", "''")
        duckdb_tool.execute_query(
            f'CREATE OR REPLACE TABLE "{table_name}" AS '
            "SELECT * FROM read_csv_auto("
            f"'{csv_sql_path}', delim='{delimiter}', header=true, "
            "sample_size=-1, all_varchar=false)"
        )
        row_count = int(
            duckdb_tool.execute_query(
                f'SELECT COUNT(*) AS n FROM "{table_name}"'
            )["n"].iloc[0]
        )
        if row_count <= 0:
            raise HTTPException(
                status_code=400,
                detail=_error_detail("CSV tidak memiliki baris data."),
            )
        if row_count > _MAX_ROW_COUNT:
            raise HTTPException(
                status_code=400,
                detail=_error_detail(
                    f"Jumlah baris ({row_count}) melebihi batas "
                    f"{_MAX_ROW_COUNT}."
                ),
            )

        profile = generate_profile_from_duckdb(duckdb_tool, table_name)
        summary = get_table_summary(duckdb_tool, table_name)
        session = _store.create(
            table_name=table_name,
            original_filename=filename,
            db_path=db_path,
            output_dir=output_dir,
            profile=profile,
            row_count=row_count,
        )
        columns = [ColumnInfo(**column) for column in summary["columns"]]
        return UploadResponse(
            session_id=session.session_id,
            table_name=table_name,
            original_filename=filename,
            row_count=row_count,
            column_count=summary["column_count"],
            columns=columns,
            detected_datetime_column=summary["detected_datetime_column"],
            numeric_column_names=summary["numeric_column_names"],
            upload_time=session.upload_time,
        )
    except HTTPException:
        _cleanup_upload(duckdb_tool, table_name, output_dir)
        raise
    except (ValueError, duckdb.Error) as exc:
        _cleanup_upload(duckdb_tool, table_name, output_dir)
        raise HTTPException(
            status_code=400,
            detail=_error_detail(f"Format CSV tidak valid: {exc}"),
        )
    except Exception as exc:  # noqa: BLE001
        _cleanup_upload(duckdb_tool, table_name, output_dir)
        raise HTTPException(
            status_code=500,
            detail=_error_detail(f"Gagal memproses upload: {exc}"),
        )
    finally:
        if duckdb_tool is not None:
            duckdb_tool.close()


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    try:
        session = _store.get_or_raise(req.session_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=_error_detail("Session tidak ditemukan."),
        )

    resolved_model = _resolve_model_tag(req.model)
    duckdb_tool = DuckDBTool(session.db_path)
    try:
        workflow = SequentialAnalyticsWorkflow(
            domain="custom",
            dataset_profile=session.profile,
            duckdb_tool=duckdb_tool,
            table_name=session.table_name,
            model_name=req.model,
        )
        state = await asyncio.to_thread(workflow.run, req.question)
    except Exception as exc:  # noqa: BLE001
        return AskResponse(
            success=False,
            session_id=req.session_id,
            question=req.question,
            answer=None,
            sql=None,
            repaired_sql=None,
            route=None,
            planned_route=None,
            route_source=None,
            latency={},
            tool_calls=[],
            error=str(exc),
            model=resolved_model,
        )
    finally:
        duckdb_tool.close()

    tool_calls = [
        ToolCallSummary(
            tool=str(call.get("tool", "")),
            status=str(call.get("status", "")),
            latency_seconds=float(call.get("latency_seconds", 0.0)),
        )
        for call in state.tool_calls
    ]
    return AskResponse(
        success=state.success,
        session_id=req.session_id,
        question=req.question,
        answer=state.final_answer,
        sql=state.generated_sql,
        repaired_sql=state.repaired_sql,
        route=state.route,
        planned_route=state.planned_route,
        route_source=state.route_source,
        latency=state.latency,
        tool_calls=tool_calls,
        error=None if state.success else state.error_message,
        model=resolved_model,
    )


@app.post("/report", response_model=ReportResponse)
async def generate_report(req: ReportRequest) -> ReportResponse:
    try:
        session = _store.get_or_raise(req.session_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=_error_detail("Session tidak ditemukan."),
        )

    resolved_model = _resolve_model_tag(req.model)
    duckdb_tool = DuckDBTool(session.db_path)
    try:
        workflow = GenericReportWorkflow(
            duckdb_tool=duckdb_tool,
            table_name=session.table_name,
            profile=session.profile,
            output_dir=session.output_dir,
            model_name=req.model,
        )
        metadata = await asyncio.to_thread(workflow.run)
    except Exception as exc:  # noqa: BLE001
        return ReportResponse(
            success=False,
            session_id=req.session_id,
            tex_success=False,
            pdf_success=False,
            chart_count=0,
            insight_success_count=0,
            insight_failed_count=0,
            pdf_download_url=None,
            latency={},
            error=str(exc),
            pdf_error=None,
            model=resolved_model,
        )
    finally:
        duckdb_tool.close()

    pdf_download_url = None
    if metadata.get("pdf_success"):
        _store.update_pdf_path(req.session_id, metadata["pdf_path"])
        pdf_download_url = f"/sessions/{req.session_id}/report.pdf"

    return ReportResponse(
        success=bool(metadata.get("success")),
        session_id=req.session_id,
        tex_success=bool(metadata.get("tex_success")),
        pdf_success=bool(metadata.get("pdf_success")),
        chart_count=int(metadata.get("chart_count", 0)),
        insight_success_count=int(metadata.get("insight_success_count", 0)),
        insight_failed_count=int(metadata.get("insight_failed_count", 0)),
        pdf_download_url=pdf_download_url,
        latency=metadata.get("latency", {}),
        error=metadata.get("error_message") or None,
        pdf_error=metadata.get("pdf_error") or None,
        model=resolved_model,
    )


@app.get("/sessions/{session_id}/report.pdf")
async def download_report(session_id: str) -> FileResponse:
    try:
        session = _store.get_or_raise(session_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=_error_detail("Session tidak ditemukan."),
        )

    if session.pdf_path is None:
        raise HTTPException(
            status_code=404, detail=_error_detail("Laporan belum dibuat.")
        )
    pdf_file = Path(session.pdf_path)
    if not pdf_file.exists():
        raise HTTPException(
            status_code=404,
            detail=_error_detail("File PDF tidak ditemukan."),
        )
    return FileResponse(
        path=str(pdf_file),
        media_type="application/pdf",
        filename=f"{session.table_name}_report.pdf",
    )


@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str) -> SessionInfo:
    try:
        session = _store.get_or_raise(session_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=_error_detail("Session tidak ditemukan."),
        )
    return SessionInfo(
        session_id=session.session_id,
        table_name=session.table_name,
        original_filename=session.original_filename,
        row_count=session.row_count,
        column_count=len(session.profile.columns),
        upload_time=session.upload_time,
        has_report=session.pdf_path is not None,
    )


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    try:
        session = _store.get_or_raise(session_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=_error_detail("Session tidak ditemukan."),
        )

    duckdb_tool = DuckDBTool(session.db_path)
    try:
        duckdb_tool.execute_query(
            f'DROP TABLE IF EXISTS "{session.table_name}"'
        )
    except Exception:  # noqa: BLE001 - best-effort
        pass
    finally:
        duckdb_tool.close()

    shutil.rmtree(session.output_dir, ignore_errors=True)
    _store.delete(session_id)
    return {"success": True, "session_id": session_id}
