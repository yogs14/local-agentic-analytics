from pathlib import Path
import sys
import threading

import pytest
from httpx import ASGITransport, AsyncClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.api import app as app_module
from local_agentic_analytics.api.app import app
from local_agentic_analytics.api.session_store import SessionStore
from local_agentic_analytics.core.dataset_profile import ColumnProfile, DatasetProfile
from local_agentic_analytics.core.state import AnalyticsState


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _dummy_profile() -> DatasetProfile:
    return DatasetProfile(
        name="dummy",
        domain="custom",
        table_name="dummy",
        datetime_column="tanggal",  # type: ignore[arg-type]
        columns={
            "tanggal": ColumnProfile(name="tanggal", type="datetime", semantic_type="datetime"),
            "nilai": ColumnProfile(name="nilai", type="numeric", semantic_type="numeric"),
        },
        sql_rules={"missing_value_count": "COUNT(*) FILTER (WHERE <column> IS NULL)"},
    )


@pytest.fixture
def csv_bytes_simple() -> bytes:
    lines = ["tanggal,produk,penjualan"]
    for i in range(10):
        lines.append(f"2024-0{(i % 9) + 1}-01,Produk {i % 3 + 1},{(i + 1) * 100.0}")
    return "\n".join(lines).encode()


@pytest.fixture
def csv_bytes_no_datetime() -> bytes:
    lines = ["produk,harga,kuantitas"]
    for i in range(5):
        lines.append(f"Produk {i},{i * 1000},{i * 10}")
    return "\n".join(lines).encode()


async def _upload(client: AsyncClient, content: bytes, filename: str = "test.csv"):
    return await client.post(
        "/upload", files={"file": (filename, content, "text/csv")}
    )


# -- Health -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_returns_200():
    async with _client() as client:
        resp = await client.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_has_required_fields():
    async with _client() as client:
        resp = await client.get("/health")
    body = resp.json()
    assert "status" in body
    assert "ollama_ok" in body
    assert "duckdb_ok" in body


# -- Upload -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_valid_csv_returns_session_id(csv_bytes_simple):
    async with _client() as client:
        resp = await _upload(client, csv_bytes_simple)
    assert resp.status_code == 200
    assert resp.json()["session_id"]


@pytest.mark.asyncio
async def test_upload_rejects_non_csv_extension():
    async with _client() as client:
        resp = await _upload(client, b"a,b,c\n1,2,3", filename="data.txt")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_upload_rejects_empty_file():
    async with _client() as client:
        resp = await _upload(client, b"   ")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_upload_returns_correct_row_count(csv_bytes_simple):
    async with _client() as client:
        resp = await _upload(client, csv_bytes_simple)
    assert resp.json()["row_count"] == 10


@pytest.mark.asyncio
async def test_upload_detects_numeric_columns(csv_bytes_simple):
    async with _client() as client:
        resp = await _upload(client, csv_bytes_simple)
    assert "penjualan" in resp.json()["numeric_column_names"]


@pytest.mark.asyncio
async def test_upload_csv_semicolon_delimiter():
    content = b"tanggal;produk;nilai\n2024-01-01;A;100\n2024-01-02;B;200"
    async with _client() as client:
        resp = await _upload(client, content)
    body = resp.json()
    assert resp.status_code == 200
    assert body["column_count"] == 3
    assert body["row_count"] == 2


# -- Ask --------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_returns_404_for_unknown_session():
    async with _client() as client:
        resp = await client.post(
            "/ask", json={"session_id": "nope", "question": "halo"}
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_ask_returns_structured_response(csv_bytes_simple, monkeypatch):
    def fake_run(self, question):
        state = AnalyticsState(user_query=question)
        state.success = True
        state.final_answer = "Total penjualan adalah 5500."
        state.generated_sql = "SELECT SUM(penjualan) FROM test"
        state.route = "llm_sql"
        state.planned_route = "STRUCTURED_SQL"
        state.route_source = "forced_energy"
        state.latency = {"total": 0.1}
        state.tool_calls = [
            {"tool": "duckdb.query", "status": "success", "latency_seconds": 0.05}
        ]
        return state

    monkeypatch.setattr(
        app_module.SequentialAnalyticsWorkflow, "run", fake_run
    )

    async with _client() as client:
        upload = await _upload(client, csv_bytes_simple)
        session_id = upload.json()["session_id"]
        resp = await client.post(
            "/ask",
            json={"session_id": session_id, "question": "Berapa total penjualan?"},
        )

    body = resp.json()
    assert resp.status_code == 200
    assert body["success"] is True
    assert body["answer"] == "Total penjualan adalah 5500."
    assert len(body["tool_calls"]) == 1


@pytest.mark.asyncio
async def test_ask_returns_success_false_on_workflow_error(
    csv_bytes_simple, monkeypatch
):
    def boom(self, question):
        raise RuntimeError("workflow exploded")

    monkeypatch.setattr(app_module.SequentialAnalyticsWorkflow, "run", boom)

    async with _client() as client:
        upload = await _upload(client, csv_bytes_simple)
        session_id = upload.json()["session_id"]
        resp = await client.post(
            "/ask", json={"session_id": session_id, "question": "Halo?"}
        )

    body = resp.json()
    assert resp.status_code == 200
    assert body["success"] is False
    assert body["error"]


# -- Report -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_report_returns_404_for_unknown_session():
    async with _client() as client:
        resp = await client.post("/report", json={"session_id": "nope"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_report_returns_structured_response(csv_bytes_simple, monkeypatch):
    def fake_run(self):
        return {
            "success": True,
            "tex_success": True,
            "pdf_success": False,
            "tex_path": "/tmp/x.tex",
            "pdf_path": "",
            "chart_count": 3,
            "insight_success_count": 3,
            "insight_failed_count": 0,
            "error_message": "",
            "pdf_error": "compiler missing",
            "latency": {},
            "tool_calls": [],
        }

    monkeypatch.setattr(app_module.GenericReportWorkflow, "run", fake_run)

    async with _client() as client:
        upload = await _upload(client, csv_bytes_simple)
        session_id = upload.json()["session_id"]
        resp = await client.post("/report", json={"session_id": session_id})

    body = resp.json()
    assert resp.status_code == 200
    assert body["tex_success"] is True
    assert body["chart_count"] == 3
    assert body["pdf_download_url"] is None


@pytest.mark.asyncio
async def test_report_pdf_download_url_set_when_pdf_success(
    csv_bytes_simple, tmp_path, monkeypatch
):
    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    def fake_run(self):
        return {
            "success": True,
            "tex_success": True,
            "pdf_success": True,
            "tex_path": "/tmp/x.tex",
            "pdf_path": str(pdf_file),
            "chart_count": 2,
            "insight_success_count": 2,
            "insight_failed_count": 0,
            "error_message": "",
            "pdf_error": "",
            "latency": {},
            "tool_calls": [],
        }

    monkeypatch.setattr(app_module.GenericReportWorkflow, "run", fake_run)

    async with _client() as client:
        upload = await _upload(client, csv_bytes_simple)
        session_id = upload.json()["session_id"]
        resp = await client.post("/report", json={"session_id": session_id})

    assert resp.json()["pdf_download_url"] == f"/sessions/{session_id}/report.pdf"


# -- Download ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_download_returns_404_before_report(csv_bytes_simple):
    async with _client() as client:
        upload = await _upload(client, csv_bytes_simple)
        session_id = upload.json()["session_id"]
        resp = await client.get(f"/sessions/{session_id}/report.pdf")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_download_serves_pdf_after_report(
    csv_bytes_simple, tmp_path, monkeypatch
):
    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy")

    def fake_run(self):
        return {
            "success": True,
            "tex_success": True,
            "pdf_success": True,
            "tex_path": "/tmp/x.tex",
            "pdf_path": str(pdf_file),
            "chart_count": 1,
            "insight_success_count": 1,
            "insight_failed_count": 0,
            "error_message": "",
            "pdf_error": "",
            "latency": {},
            "tool_calls": [],
        }

    monkeypatch.setattr(app_module.GenericReportWorkflow, "run", fake_run)

    async with _client() as client:
        upload = await _upload(client, csv_bytes_simple)
        session_id = upload.json()["session_id"]
        await client.post("/report", json={"session_id": session_id})
        resp = await client.get(f"/sessions/{session_id}/report.pdf")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"


# -- Session management -----------------------------------------------------

@pytest.mark.asyncio
async def test_get_session_returns_info(csv_bytes_simple):
    async with _client() as client:
        upload = await _upload(client, csv_bytes_simple)
        session_id = upload.json()["session_id"]
        resp = await client.get(f"/sessions/{session_id}")
    body = resp.json()
    assert resp.status_code == 200
    assert body["session_id"] == session_id
    assert body["has_report"] is False


@pytest.mark.asyncio
async def test_delete_session_removes_it(csv_bytes_simple):
    async with _client() as client:
        upload = await _upload(client, csv_bytes_simple)
        session_id = upload.json()["session_id"]
        resp = await client.delete(f"/sessions/{session_id}")
        assert resp.status_code == 200
        follow = await client.get(f"/sessions/{session_id}")
    assert follow.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_session_returns_404():
    async with _client() as client:
        resp = await client.delete("/sessions/does-not-exist")
    assert resp.status_code == 404


# -- SessionStore unit tests ------------------------------------------------

def _store_create(store: SessionStore):
    return store.create(
        table_name="t",
        original_filename="t.csv",
        db_path=":memory:",
        output_dir=Path("/tmp/x"),
        profile=_dummy_profile(),
        row_count=5,
    )


def test_store_create_and_get():
    store = SessionStore()
    session = _store_create(store)
    assert store.get(session.session_id) is session


def test_store_get_nonexistent_returns_none():
    store = SessionStore()
    assert store.get("missing") is None


def test_store_get_or_raise_raises_key_error():
    store = SessionStore()
    with pytest.raises(KeyError):
        store.get_or_raise("missing")


def test_store_delete_returns_false_unknown():
    store = SessionStore()
    assert store.delete("missing") is False


def test_store_update_pdf_path():
    store = SessionStore()
    session = _store_create(store)
    store.update_pdf_path(session.session_id, "/tmp/x/report.pdf")
    assert store.get(session.session_id).pdf_path == "/tmp/x/report.pdf"


def test_session_store_thread_safety():
    store = SessionStore()

    def worker():
        _store_create(store)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert store.count() == 50
