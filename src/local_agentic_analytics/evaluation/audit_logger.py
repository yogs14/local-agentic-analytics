"""Lightweight JSONL audit logging for tool calls."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from local_agentic_analytics.core.config import PROJECT_ROOT


DEFAULT_AUDIT_LOG_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "tool_call_audit.jsonl"
)


@dataclass
class ToolCallAuditEvent:
    timestamp: str
    component: str
    action: str
    tool: str
    status: str
    latency_seconds: float
    input_summary: str = ""
    output_summary: str = ""
    error_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolAuditLogger:
    def __init__(self, log_path: str | Path = DEFAULT_AUDIT_LOG_PATH):
        self.log_path = Path(log_path)

    def log_event(self, event: ToolCallAuditEvent) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        except Exception:
            # Audit logging must never break the user-facing workflow.
            return

    def log(
        self,
        component: str,
        action: str,
        status: str,
        latency_seconds: float,
        tool: str | None = None,
        input_summary: str = "",
        output_summary: str = "",
        error_message: str = "",
        metadata: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        event = ToolCallAuditEvent(
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            component=component,
            action=action,
            tool=tool or f"{component}.{action}",
            status=status,
            latency_seconds=round(float(latency_seconds), 6),
            input_summary=_truncate(input_summary),
            output_summary=_truncate(output_summary),
            error_message=_truncate(error_message),
            metadata=_json_safe_metadata(metadata or {}),
        )
        self.log_event(event)
        return asdict(event)


def summarize_value(value: Any) -> str:
    if value is None:
        return ""

    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"{type(value).__name__}(shape={shape})"

    if isinstance(value, dict):
        keys = ", ".join(str(key) for key in list(value.keys())[:8])
        return f"dict(keys=[{keys}])"

    text = str(value)
    return _truncate(text)


def _truncate(text: str, max_length: int = 500) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _json_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        json.dumps(metadata)
        return metadata
    except TypeError:
        return {"repr": _truncate(str(metadata))}
