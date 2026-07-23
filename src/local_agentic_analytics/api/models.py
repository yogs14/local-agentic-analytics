"""Pydantic models untuk API request dan response."""
from __future__ import annotations

from pydantic import BaseModel, Field

# Tag Ollama seperti "qwen2.5:1.5b" atau "gemma2-energy-insight:v3" —
# huruf/angka/titik/titik-dua/strip/underscore, sesuai konvensi penamaan model
# Ollama (lihat `ollama list`).
_MODEL_TAG_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,127}$"


# -- Request Models ---------------------------------------------------------

class AskRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=1, max_length=2000)
    engine: str = Field(default="custom", pattern="^(custom|langgraph)$")
    model: str | None = Field(
        default=None,
        pattern=_MODEL_TAG_PATTERN,
        description="Tag model Ollama override, mis. 'qwen2.5:1.5b'. Kosong = default server (OLLAMA_MODEL).",
    )


class ReportRequest(BaseModel):
    session_id: str
    engine: str = Field(default="custom", pattern="^(custom|langgraph)$")
    model: str | None = Field(
        default=None,
        pattern=_MODEL_TAG_PATTERN,
        description="Tag model Ollama override untuk narasi insight laporan.",
    )


# -- Response Models --------------------------------------------------------

class ColumnInfo(BaseModel):
    name: str
    duckdb_type: str
    semantic_type: str  # numeric/integer/categorical/datetime/boolean/other
    is_datetime: bool
    is_numeric: bool


class UploadResponse(BaseModel):
    success: bool = True
    session_id: str
    table_name: str
    original_filename: str
    row_count: int
    column_count: int
    columns: list[ColumnInfo]
    detected_datetime_column: str | None
    numeric_column_names: list[str]
    upload_time: str


class ToolCallSummary(BaseModel):
    tool: str
    status: str
    latency_seconds: float


class AskResponse(BaseModel):
    success: bool
    session_id: str
    question: str
    answer: str | None
    sql: str | None
    repaired_sql: str | None
    route: str | None
    planned_route: str | None
    route_source: str | None
    latency: dict[str, float]
    tool_calls: list[ToolCallSummary]
    error: str | None
    model: str


class ReportResponse(BaseModel):
    success: bool
    session_id: str
    tex_success: bool
    pdf_success: bool
    chart_count: int
    insight_success_count: int
    insight_failed_count: int
    pdf_download_url: str | None
    latency: dict[str, float]
    error: str | None
    pdf_error: str | None
    model: str


class ModelInfo(BaseModel):
    key: str
    ollama_tag: str
    param_count: str
    quantization: str
    is_default: bool
    installed: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    default_model: str


class SessionInfo(BaseModel):
    session_id: str
    table_name: str
    original_filename: str
    row_count: int
    column_count: int
    upload_time: str
    has_report: bool


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    ollama_ok: bool
    duckdb_ok: bool


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
