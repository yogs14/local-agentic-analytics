"""Import shim for running ``python -m local_agentic_analytics`` from the repo root."""

from pathlib import Path


_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "local_agentic_analytics"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))
