from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.audit_logger import (
    ToolAuditLogger,
    summarize_value,
)


def test_tool_audit_logger_writes_jsonl_event(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    logger = ToolAuditLogger(log_path=log_path)

    logger.log(
        component="DuckDBTool",
        action="execute_query",
        tool="duckdb.query",
        status="success",
        latency_seconds=0.1234567,
        input_summary="SELECT 1",
        output_summary="DataFrame(shape=(1, 1))",
        metadata={"row_count": 1},
    )

    rows = log_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    event = json.loads(rows[0])
    assert event["component"] == "DuckDBTool"
    assert event["action"] == "execute_query"
    assert event["tool"] == "duckdb.query"
    assert event["status"] == "success"
    assert event["latency_seconds"] == 0.123457
    assert event["metadata"] == {"row_count": 1}


def test_summarize_value_handles_dict_and_long_text():
    assert summarize_value({"a": 1, "b": 2}) == "dict(keys=[a, b])"
    assert summarize_value("x" * 600).endswith("...")
