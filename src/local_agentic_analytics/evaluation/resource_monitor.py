"""Resource sampling and throughput helpers for GPU/CPU benchmarking.

This module is measurement-only: it never changes model configuration or
workflow behavior. VRAM sampling degrades gracefully to a no-op when
``nvidia-smi`` is not available.
"""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Callable

import psutil


NVIDIA_SMI_VRAM_COMMAND = (
    "nvidia-smi",
    "--query-gpu=memory.used",
    "--format=csv,noheader,nounits",
)
NVIDIA_SMI_COMPUTE_APPS_COMMAND = (
    "nvidia-smi",
    "--query-compute-apps=pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
)
DEFAULT_VRAM_PROCESS_SUBSTRING = "ollama"
DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.1

VramReader = Callable[[], "float | None"]
RssReader = Callable[[], "float | None"]


@dataclass(frozen=True)
class ResourceUsage:
    """Peak resource usage captured across a sampling window."""

    peak_vram_mb: float | None
    peak_rss_mb: float | None
    vram_available: bool
    sample_count: int


def query_gpu_vram_mb(
    runner: Callable[[], str] | None = None,
) -> float | None:
    """Read currently used GPU VRAM in MB via ``nvidia-smi``.

    Returns ``None`` when ``nvidia-smi`` is absent or its output cannot be
    parsed, so callers can treat a missing GPU as a graceful no-op.
    """
    runner = runner or _run_nvidia_smi_vram
    try:
        output = runner()
    except FileNotFoundError:
        return None
    except Exception:
        return None

    return _parse_first_vram_value(output)


def _run_nvidia_smi_vram() -> str:
    result = subprocess.run(
        list(NVIDIA_SMI_VRAM_COMMAND),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _parse_first_vram_value(output: str) -> float | None:
    if not output:
        return None
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def query_ollama_vram_mb(
    runner: Callable[[], str] | None = None,
    process_substring: str = DEFAULT_VRAM_PROCESS_SUBSTRING,
) -> float | None:
    """Sum VRAM (MB) used by Ollama compute processes only.

    Reads per-process GPU memory via
    ``nvidia-smi --query-compute-apps`` and sums ``used_memory`` for processes
    whose name contains ``process_substring`` (case-insensitive). This excludes
    the desktop, browser, and models left resident by other workloads.

    Returns the attributable sum (``0.0`` when no matching process is on the
    GPU, e.g. CPU-only inference). Returns ``None`` only when the query itself
    is unavailable, so callers can fall back gracefully.
    """
    runner = runner or _run_nvidia_smi_compute_apps
    try:
        output = runner()
    except FileNotFoundError:
        return None
    except Exception:
        return None

    return parse_compute_apps_vram_mb(output, process_substring)


def _run_nvidia_smi_compute_apps() -> str:
    result = subprocess.run(
        list(NVIDIA_SMI_COMPUTE_APPS_COMMAND),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_compute_apps_vram_mb(
    output: str,
    process_substring: str = DEFAULT_VRAM_PROCESS_SUBSTRING,
) -> float:
    """Sum ``used_memory`` for compute-app rows whose name matches the filter."""
    needle = process_substring.lower()
    total = 0.0
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) < 3:
            continue
        process_name = parts[1].lower()
        if needle not in process_name:
            continue
        memory_mb = _safe_float(parts[2])
        if memory_mb is not None and memory_mb > 0:
            total += memory_mb
    return total


def make_global_delta_vram_reader(
    global_reader: Callable[[], "float | None"] | None = None,
) -> "VramReader":
    """Build a fallback reader returning the baseline->peak global VRAM delta.

    The first sample captures the baseline (global ``memory.used``); subsequent
    samples report ``current - baseline`` (clamped at 0). Used only when
    per-process attribution is unavailable.
    """
    reader = global_reader or query_gpu_vram_mb
    baseline: list[float | None] = [None]

    def delta_reader() -> float | None:
        current = reader()
        if current is None:
            return None
        if baseline[0] is None:
            baseline[0] = current
            return 0.0
        return max(0.0, current - baseline[0])

    return delta_reader


def _safe_float(text: Any) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _read_process_rss_mb() -> float | None:
    try:
        rss_bytes = psutil.Process().memory_info().rss
    except Exception:
        return None
    return rss_bytes / (1024 * 1024)


class ResourceSampler:
    """Sample peak GPU VRAM and process RSS from a background thread."""

    def __init__(
        self,
        interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
        vram_reader: VramReader | None = None,
        rss_reader: RssReader | None = None,
    ):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than 0")

        self.interval_seconds = interval_seconds
        self._vram_reader = vram_reader or query_gpu_vram_mb
        self._rss_reader = rss_reader or _read_process_rss_mb
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_vram_mb: float | None = None
        self._peak_rss_mb: float | None = None
        self._vram_available = False
        self._sample_count = 0

    def __enter__(self) -> "ResourceSampler":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("ResourceSampler is already running")
        # Take one immediate sample so very short windows still record usage.
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceUsage:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()
            self._thread = None
        return ResourceUsage(
            peak_vram_mb=self._peak_vram_mb,
            peak_rss_mb=self._peak_rss_mb,
            vram_available=self._vram_available,
            sample_count=self._sample_count,
        )

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample_once()

    def _sample_once(self) -> None:
        self._sample_count += 1

        vram = self._vram_reader()
        if vram is not None:
            self._vram_available = True
            if self._peak_vram_mb is None or vram > self._peak_vram_mb:
                self._peak_vram_mb = vram

        rss = self._rss_reader()
        if rss is not None:
            if self._peak_rss_mb is None or rss > self._peak_rss_mb:
                self._peak_rss_mb = rss


def tokens_per_second(
    eval_count: Any,
    eval_duration: Any,
) -> float | None:
    """Compute generation throughput as ``eval_count / eval_duration``.

    ``eval_duration`` is the decode time only (Ollama excludes load and prompt
    evaluation from it), so load time never pollutes this figure. Returns
    ``None`` when inputs are missing, non-numeric, or non-positive.
    """
    count = _coerce_positive_float(eval_count)
    duration = _coerce_positive_float(eval_duration)
    if count is None or duration is None:
        return None
    return count / duration


def aggregate_tokens_per_second(
    tool_calls: list[dict[str, Any]],
) -> float | None:
    """Aggregate TPS across the Ollama generations recorded for one run.

    Sums ``eval_count`` and ``eval_duration`` from the metadata of every
    ``ollama`` tool call so the result reflects total decoded tokens over total
    decode time. ``load_duration`` is intentionally never consulted.
    """
    total_eval_count = 0.0
    total_eval_duration = 0.0
    for event in tool_calls:
        if not isinstance(event, dict):
            continue
        if event.get("component") != "ollama":
            continue
        metadata = event.get("metadata")
        if not isinstance(metadata, dict):
            continue
        count = _coerce_positive_float(metadata.get("eval_count"))
        duration = _coerce_positive_float(metadata.get("eval_duration"))
        if count is None or duration is None:
            continue
        total_eval_count += count
        total_eval_duration += duration

    if total_eval_duration <= 0:
        return None
    return total_eval_count / total_eval_duration


def _coerce_positive_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str) and value.strip():
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    if number <= 0:
        return None
    return number
