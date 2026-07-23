"""In-memory session store untuk CSV upload sessions."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from local_agentic_analytics.core.dataset_profile import DatasetProfile


@dataclass
class UploadSession:
    """Satu session CSV upload."""

    session_id: str
    table_name: str
    original_filename: str
    db_path: str
    output_dir: Path
    profile: DatasetProfile
    row_count: int
    upload_time: str  # ISO 8601
    pdf_path: str | None = None  # diisi setelah report berhasil


class SessionStore:
    """Thread-safe in-memory store untuk :class:`UploadSession`.

    Tidak ada persistence - data hilang saat proses restart. Didesain untuk
    single-server, single-process deployment.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, UploadSession] = {}
        self._lock = threading.Lock()

    def create(
        self,
        table_name: str,
        original_filename: str,
        db_path: str,
        output_dir: Path,
        profile: DatasetProfile,
        row_count: int,
    ) -> UploadSession:
        """Buat session baru dan simpan. Return session yang dibuat."""
        session = UploadSession(
            session_id=str(uuid.uuid4()),
            table_name=table_name,
            original_filename=original_filename,
            db_path=db_path,
            output_dir=Path(output_dir),
            profile=profile,
            row_count=row_count,
            upload_time=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> UploadSession | None:
        """Return session atau None jika tidak ada."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_raise(self, session_id: str) -> UploadSession:
        """Return session atau raise KeyError dengan pesan jelas."""
        session = self.get(session_id)
        if session is None:
            raise KeyError(f"Session tidak ditemukan: {session_id}")
        return session

    def update_pdf_path(self, session_id: str, pdf_path: str) -> None:
        """Update pdf_path setelah report berhasil."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.pdf_path = pdf_path

    def delete(self, session_id: str) -> bool:
        """Hapus session. Return True jika ada, False jika tidak."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list_all(self) -> list[UploadSession]:
        """Return semua session aktif (untuk debugging)."""
        with self._lock:
            return list(self._sessions.values())

    def count(self) -> int:
        """Jumlah session aktif."""
        with self._lock:
            return len(self._sessions)
