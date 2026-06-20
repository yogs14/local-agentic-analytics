from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.evaluation.gpu_cpu_benchmark import (
    GPU_REQUESTED_BUT_RAN_ON_CPU,
    BenchmarkMode,
    GpuEngagement,
    aggregate_repeated_summaries,
    aggregate_stats,
    default_modes,
    detect_gpu_engagement,
    is_oom_error,
    parse_ollama_ps,
    parse_ollama_ps_vram,
    read_ollama_ps_vram,
    run_gpu_cpu_benchmark,
    run_gpu_cpu_benchmark_repeated,
)
from local_agentic_analytics.evaluation.resource_monitor import (
    ResourceSampler,
    ResourceUsage,
    aggregate_tokens_per_second,
    make_global_delta_vram_reader,
    parse_compute_apps_vram_mb,
    query_gpu_vram_mb,
    query_ollama_vram_mb,
    tokens_per_second,
)
from local_agentic_analytics.tools.ollama_tool import OllamaTool


# --- Tokens-per-second ---------------------------------------------------------


def test_tokens_per_second_basic():
    assert tokens_per_second(20, 2.0) == 10.0


def test_tokens_per_second_rejects_invalid_or_zero():
    assert tokens_per_second(20, 0) is None
    assert tokens_per_second(None, 2.0) is None
    assert tokens_per_second(20, None) is None
    assert tokens_per_second(0, 2.0) is None
    assert tokens_per_second("not-a-number", 2.0) is None


def test_aggregate_tokens_per_second_sums_only_ollama_eval_metrics():
    tool_calls = [
        {
            "component": "ollama",
            "metadata": {
                "eval_count": 20,
                "eval_duration": 2.0,
                "load_duration": 99.0,  # must be ignored
            },
        },
        {
            "component": "ollama",
            "metadata": {"eval_count": 10, "eval_duration": 1.0},
        },
        {
            "component": "duckdb",
            "metadata": {"eval_count": 999, "eval_duration": 0.001},
        },
    ]

    # (20 + 10) / (2.0 + 1.0) = 10.0; the duckdb event and load_duration excluded.
    assert aggregate_tokens_per_second(tool_calls) == 10.0


def test_aggregate_tokens_per_second_returns_none_without_metrics():
    assert aggregate_tokens_per_second([]) is None
    assert aggregate_tokens_per_second(
        [{"component": "ollama", "metadata": {}}]
    ) is None


# --- VRAM query and graceful degradation ---------------------------------------


def test_query_gpu_vram_mb_parses_first_value():
    assert query_gpu_vram_mb(runner=lambda: "2097\n") == 2097.0


def test_query_gpu_vram_mb_none_when_nvidia_smi_absent():
    def missing_runner():
        raise FileNotFoundError("nvidia-smi")

    assert query_gpu_vram_mb(runner=missing_runner) is None


def test_query_gpu_vram_mb_none_on_unparseable_output():
    assert query_gpu_vram_mb(runner=lambda: "no-gpu-here") is None


# --- Per-process (Ollama-attributable) VRAM ------------------------------------


COMPUTE_APPS_OUTPUT = (
    "1234, ollama.exe, 1900\n"
    "5678, chrome.exe, 800\n"
    "9012, ollama_llama_server, 100\n"
    "3456, gnome-shell, 250\n"
)


def test_parse_compute_apps_sums_only_ollama_processes():
    # 1900 (ollama.exe) + 100 (ollama_llama_server); chrome/gnome excluded.
    assert parse_compute_apps_vram_mb(COMPUTE_APPS_OUTPUT) == 2000.0


def test_parse_compute_apps_ignores_non_ollama_processes():
    output = "5678, chrome.exe, 800\n3456, gnome-shell, 250\n"

    assert parse_compute_apps_vram_mb(output) == 0.0


def test_query_ollama_vram_is_zero_when_ollama_not_on_gpu():
    # nvidia-smi works but no Ollama process is resident -> attributable 0.
    output = "5678, chrome.exe, 800\n"

    assert query_ollama_vram_mb(runner=lambda: output) == 0.0


def test_query_ollama_vram_sums_matching_processes():
    assert query_ollama_vram_mb(runner=lambda: COMPUTE_APPS_OUTPUT) == 2000.0


def test_query_ollama_vram_none_when_nvidia_smi_absent():
    def missing_runner():
        raise FileNotFoundError("nvidia-smi")

    assert query_ollama_vram_mb(runner=missing_runner) is None


def test_global_delta_vram_reader_reports_increase_over_baseline():
    readings = iter([3000.0, 3000.0, 4200.0])
    reader = make_global_delta_vram_reader(global_reader=lambda: next(readings))

    assert reader() == 0.0  # baseline captured
    assert reader() == 0.0  # unchanged
    assert reader() == 1200.0  # delta over baseline


def test_global_delta_vram_reader_none_when_global_unavailable():
    reader = make_global_delta_vram_reader(global_reader=lambda: None)

    assert reader() is None


def test_resource_sampler_degrades_when_no_gpu():
    sampler = ResourceSampler(
        interval_seconds=10,
        vram_reader=lambda: None,  # simulates absent GPU
        rss_reader=lambda: 250.0,
    )

    sampler.start()
    usage = sampler.stop()

    assert usage.vram_available is False
    assert usage.peak_vram_mb is None
    assert usage.peak_rss_mb == 250.0
    assert usage.sample_count >= 1


def test_resource_sampler_tracks_peak_vram_and_rss():
    vram_values = iter([1000.0, 2500.0, 1500.0])
    rss_values = iter([100.0, 400.0, 200.0])
    sampler = ResourceSampler(
        interval_seconds=0.01,
        vram_reader=lambda: next(vram_values, 0.0),
        rss_reader=lambda: next(rss_values, 0.0),
    )

    sampler.start()
    # Let the background thread take a couple more samples deterministically.
    for _ in range(2):
        sampler._sample_once()
    usage = sampler.stop()

    assert usage.vram_available is True
    assert usage.peak_vram_mb == 2500.0
    assert usage.peak_rss_mb == 400.0


# --- ollama ps parsing ---------------------------------------------------------


PS_HEADER = "NAME         ID              SIZE      PROCESSOR    CONTEXT    UNTIL"


def test_parse_ollama_ps_full_gpu():
    output = (
        f"{PS_HEADER}\n"
        "gemma2:2b    8ccf136fdd52    1.9 GB    100% GPU     4096       2 minutes"
    )

    engagement = parse_ollama_ps(output, "gemma2:2b")

    assert engagement.on_gpu is True
    assert engagement.partial_offload is False
    assert engagement.processor == "100% GPU"


def test_parse_ollama_ps_partial_offload():
    output = (
        f"{PS_HEADER}\n"
        "gemma2:2b    8ccf136fdd52    1.9 GB    48%/52% CPU/GPU   4096    2 minutes"
    )

    engagement = parse_ollama_ps(output, "gemma2:2b")

    assert engagement.on_gpu is True
    assert engagement.partial_offload is True
    assert "CPU/GPU" in engagement.processor


def test_parse_ollama_ps_cpu_only():
    output = (
        f"{PS_HEADER}\n"
        "gemma2:2b    8ccf136fdd52    1.9 GB    100% CPU     4096       2 minutes"
    )

    engagement = parse_ollama_ps(output, "gemma2:2b")

    assert engagement.on_gpu is False
    assert engagement.partial_offload is False


def test_parse_ollama_ps_model_missing():
    engagement = parse_ollama_ps(f"{PS_HEADER}\n", "gemma2:2b")

    assert engagement.on_gpu is False
    assert "not found" in engagement.detail


def test_detect_gpu_engagement_when_ollama_absent():
    def missing_runner():
        raise FileNotFoundError("ollama")

    engagement = detect_gpu_engagement("gemma2:2b", ps_runner=missing_runner)

    assert engagement.on_gpu is False
    assert "not found" in engagement.detail


def test_is_oom_error():
    assert is_oom_error("CUDA error: out of memory") is True
    assert is_oom_error("cudaMalloc failed") is True
    assert is_oom_error("column not found") is False
    assert is_oom_error(None) is False


# --- Benchmark runner ----------------------------------------------------------


def _make_state(query: str, success: bool = True) -> AnalyticsState:
    state = AnalyticsState(user_query=query, success=success)
    state.latency["total"] = 1.0
    state.tool_calls = [
        {
            "component": "ollama",
            "metadata": {
                "eval_count": 20,
                "eval_duration": 2.0,
                "load_duration": 50.0,
            },
        }
    ]
    return state


class FakeWorkflow:
    def __init__(self, num_gpu: int, errors: dict[str, str] | None = None):
        self.num_gpu = num_gpu
        self.errors = errors or {}
        self.calls: list[str] = []

    def run(self, query: str) -> AnalyticsState:
        self.calls.append(query)
        if query in self.errors:
            raise RuntimeError(self.errors[query])
        return _make_state(query)


class FakeSampler:
    def __init__(self, usage: ResourceUsage):
        self.usage = usage
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> ResourceUsage:
        self.stopped = True
        return self.usage


def _build_factory(created: list[FakeWorkflow], errors: dict[str, str] | None = None):
    def factory(num_gpu: int) -> FakeWorkflow:
        workflow = FakeWorkflow(num_gpu=num_gpu, errors=errors)
        created.append(workflow)
        return workflow

    return factory


def _sampler_factory():
    return FakeSampler(
        ResourceUsage(
            peak_vram_mb=2100.0,
            peak_rss_mb=500.0,
            vram_available=True,
            sample_count=3,
        )
    )


QUESTIONS = [
    {"id": "Q1", "question": "Pertanyaan satu?"},
    {"id": "Q2", "question": "Pertanyaan dua?"},
]

PS_VRAM_HEADER = "NAME       ID    SIZE    PROCESSOR        CONTEXT   UNTIL"
GPU_PS_OUTPUT = f"{PS_VRAM_HEADER}\ngemma2:2b  abc   1.9 GB  100% GPU         4096      2 minutes\n"
CPU_PS_OUTPUT = f"{PS_VRAM_HEADER}\ngemma2:2b  abc   1.9 GB  100% CPU         4096      2 minutes\n"
SPLIT_PS_OUTPUT = f"{PS_VRAM_HEADER}\ngemma2:2b  abc   1.9 GB  30%/70% CPU/GPU  4096      2 minutes\n"

# 1.9 GB * 1024 MB/GB.
GPU_SIZE_MB = 1.9 * 1024.0


def _gpu_runners() -> dict:
    """Default injected runners for a GPU-resident benchmark run."""
    return {
        "ps_vram_runner": lambda: GPU_PS_OUTPUT,
        "global_vram_reader": lambda: 1000.0,
        "compute_apps_runner": lambda: "",
    }


def test_run_records_both_modes_and_excludes_warmup():
    created: list[FakeWorkflow] = []
    rows, summary = run_gpu_cpu_benchmark(
        QUESTIONS,
        workflow_factory=_build_factory(created),
        gpu_detector=lambda model: GpuEngagement(
            on_gpu=True, partial_offload=False, processor="100% GPU"
        ),
        sampler_factory=_sampler_factory,
        unloader=lambda model: None,
        warmup_question="warmup",
        logger=lambda message: None,
        **_gpu_runners(),
    )

    assert {row["mode"] for row in rows} == {"gpu", "cpu"}
    assert len(rows) == 4  # 2 questions x 2 modes (warm-up excluded)

    # num_gpu overridden per mode; warm-up adds exactly one extra call each.
    gpu_workflow, cpu_workflow = created
    assert gpu_workflow.num_gpu == default_modes()[0].num_gpu
    assert cpu_workflow.num_gpu == 0
    assert len(gpu_workflow.calls) == len(QUESTIONS) + 1
    assert gpu_workflow.calls[0] == "warmup"

    assert summary["modes"]["gpu"]["label"] == "gpu"
    assert summary["modes"]["gpu"]["avg_tokens_per_second"] == 10.0
    # Primary VRAM is from ollama ps; peak_global is the cross-check sampler peak.
    assert summary["modes"]["gpu"]["vram_mb"] == GPU_SIZE_MB
    assert summary["modes"]["gpu"]["processor_split"] == "100% GPU"
    assert summary["modes"]["gpu"]["peak_global_vram_mb"] == 2100.0
    assert summary["modes"]["gpu"]["baseline_global_vram_mb"] == 1000.0
    assert summary["modes"]["gpu"]["vram_delta_mb"] == 1100.0
    assert summary["modes"]["gpu"]["peak_rss_mb"] == 500.0
    assert summary["modes"]["gpu"]["fit_in_4gb"] is True
    assert summary["modes"]["cpu"]["fit_in_4gb"] is True
    assert summary["modes"]["cpu"]["success_rate"] == 1.0


def test_gpu_requested_but_ran_on_cpu_is_relabeled():
    created: list[FakeWorkflow] = []
    logs: list[str] = []
    rows, summary = run_gpu_cpu_benchmark(
        QUESTIONS,
        workflow_factory=_build_factory(created),
        gpu_detector=lambda model: GpuEngagement(
            on_gpu=False, partial_offload=False, processor="100% CPU"
        ),
        sampler_factory=_sampler_factory,
        unloader=lambda model: None,
        ps_vram_runner=lambda: CPU_PS_OUTPUT,
        global_vram_reader=lambda: 1000.0,
        compute_apps_runner=lambda: "",
        warmup_question="warmup",
        logger=logs.append,
    )

    gpu_rows = [row for row in rows if row["mode"] == "gpu"]
    assert all(row["label"] == GPU_REQUESTED_BUT_RAN_ON_CPU for row in gpu_rows)
    assert summary["modes"]["gpu"]["label"] == GPU_REQUESTED_BUT_RAN_ON_CPU
    assert summary["modes"]["gpu"]["fit_in_4gb"] is False
    # Ran on CPU -> no model-attributable VRAM.
    assert summary["modes"]["gpu"]["vram_mb"] == 0.0
    assert summary["modes"]["gpu"]["processor_split"] == "100% CPU"
    assert any(GPU_REQUESTED_BUT_RAN_ON_CPU in message for message in logs)


def test_oom_is_caught_and_run_continues():
    created: list[FakeWorkflow] = []
    logs: list[str] = []
    rows, summary = run_gpu_cpu_benchmark(
        QUESTIONS,
        workflow_factory=_build_factory(
            created,
            errors={"Pertanyaan satu?": "CUDA error: out of memory"},
        ),
        gpu_detector=lambda model: GpuEngagement(
            on_gpu=True, partial_offload=False, processor="100% GPU"
        ),
        sampler_factory=_sampler_factory,
        unloader=lambda model: None,
        warmup_question="warmup",
        logger=logs.append,
        **_gpu_runners(),
    )

    gpu_rows = {row["question_id"]: row for row in rows if row["mode"] == "gpu"}
    assert gpu_rows["Q1"]["oom"] is True
    assert gpu_rows["Q1"]["success"] is False
    # The second question still ran after the OOM.
    assert gpu_rows["Q2"]["success"] is True
    assert summary["modes"]["gpu"]["oom_count"] == 1
    assert summary["modes"]["gpu"]["fit_in_4gb"] is False


def test_partial_offload_marks_not_fit():
    created: list[FakeWorkflow] = []
    _rows, summary = run_gpu_cpu_benchmark(
        QUESTIONS,
        workflow_factory=_build_factory(created),
        gpu_detector=lambda model: GpuEngagement(
            on_gpu=True, partial_offload=True, processor="48%/52% CPU/GPU"
        ),
        sampler_factory=_sampler_factory,
        unloader=lambda model: None,
        ps_vram_runner=lambda: SPLIT_PS_OUTPUT,
        global_vram_reader=lambda: 1000.0,
        compute_apps_runner=lambda: "",
        warmup_question="warmup",
        logger=lambda message: None,
    )

    assert summary["modes"]["gpu"]["fit_in_4gb"] is False
    assert summary["modes"]["gpu"]["label"] == "gpu"
    # 70% of the model's layers are on the GPU.
    assert summary["modes"]["gpu"]["processor_split"] == "30%/70% CPU/GPU"
    assert summary["modes"]["gpu"]["vram_mb"] == GPU_SIZE_MB * 0.7


def test_unloader_invoked_between_modes():
    created: list[FakeWorkflow] = []
    unload_calls: list[str] = []

    run_gpu_cpu_benchmark(
        QUESTIONS,
        workflow_factory=_build_factory(created),
        gpu_detector=lambda model: GpuEngagement(
            on_gpu=True, partial_offload=False, processor="100% GPU"
        ),
        sampler_factory=_sampler_factory,
        unloader=unload_calls.append,
        warmup_question="warmup",
        logger=lambda message: None,
        **_gpu_runners(),
    )

    # Unload runs once per mode (clean GPU before each), including before CPU.
    assert len(unload_calls) == len(default_modes())


# --- ollama ps VRAM parsing (WDDM-safe primary source) -------------------------


def test_parse_ollama_ps_vram_full_gpu():
    result = parse_ollama_ps_vram(GPU_PS_OUTPUT, "gemma2:2b")

    assert result.found is True
    assert result.size_mb == GPU_SIZE_MB
    assert result.gpu_fraction == 1.0
    assert result.vram_mb == GPU_SIZE_MB
    assert result.processor == "100% GPU"


def test_parse_ollama_ps_vram_cpu_only():
    result = parse_ollama_ps_vram(CPU_PS_OUTPUT, "gemma2:2b")

    assert result.found is True
    assert result.gpu_fraction == 0.0
    assert result.vram_mb == 0.0
    assert result.processor == "100% CPU"


def test_parse_ollama_ps_vram_split():
    result = parse_ollama_ps_vram(SPLIT_PS_OUTPUT, "gemma2:2b")

    assert result.gpu_fraction == 0.7
    assert result.vram_mb == GPU_SIZE_MB * 0.7
    assert result.processor == "30%/70% CPU/GPU"


def test_parse_ollama_ps_vram_model_not_found():
    result = parse_ollama_ps_vram(f"{PS_VRAM_HEADER}\n", "gemma2:2b")

    assert result.found is False
    assert result.vram_mb == 0.0
    assert "not found" in result.detail


def test_parse_ollama_ps_vram_does_not_misread_model_tag_as_size():
    # "gemma2:2b" must not be parsed as a 2-byte SIZE; SIZE is "1.9 GB".
    result = parse_ollama_ps_vram(GPU_PS_OUTPUT, "gemma2:2b")

    assert result.size_mb == GPU_SIZE_MB


def test_read_ollama_ps_vram_graceful_when_ollama_absent():
    def missing_runner():
        raise FileNotFoundError("ollama")

    result = read_ollama_ps_vram("gemma2:2b", ps_runner=missing_runner)

    assert result.found is False
    assert result.vram_mb == 0.0


def test_gpu_mode_vram_mb_from_ollama_ps():
    created: list[FakeWorkflow] = []
    _rows, summary = run_gpu_cpu_benchmark(
        QUESTIONS,
        modes=(BenchmarkMode(name="gpu", num_gpu=999, requires_gpu=True),),
        workflow_factory=_build_factory(created),
        gpu_detector=lambda model: GpuEngagement(
            on_gpu=True, partial_offload=False, processor="100% GPU"
        ),
        sampler_factory=_sampler_factory,
        unloader=lambda model: None,
        ps_vram_runner=lambda: GPU_PS_OUTPUT,
        global_vram_reader=lambda: 1000.0,
        compute_apps_runner=lambda: "",
        warmup_question="warmup",
        logger=lambda message: None,
    )

    gpu = summary["modes"]["gpu"]
    assert gpu["vram_mb"] == GPU_SIZE_MB  # non-zero, under 4 GB
    assert gpu["vram_mb"] < 4096
    assert gpu["processor_split"] == "100% GPU"


def test_cpu_mode_vram_zero_from_ollama_ps():
    created: list[FakeWorkflow] = []
    _rows, summary = run_gpu_cpu_benchmark(
        QUESTIONS,
        modes=(BenchmarkMode(name="cpu", num_gpu=0, requires_gpu=False),),
        workflow_factory=_build_factory(created),
        sampler_factory=_sampler_factory,
        unloader=lambda model: None,
        ps_vram_runner=lambda: CPU_PS_OUTPUT,
        global_vram_reader=lambda: 1000.0,
        compute_apps_runner=lambda: "",
        warmup_question="warmup",
        logger=lambda message: None,
    )

    cpu = summary["modes"]["cpu"]
    assert cpu["vram_mb"] == 0.0
    assert cpu["processor_split"] == "100% CPU"


def test_vram_delta_cross_check_is_peak_minus_baseline():
    # baseline 1500 (after unload), sampler peak 2100 -> delta 600.
    def sampler_factory() -> FakeSampler:
        return FakeSampler(
            ResourceUsage(
                peak_vram_mb=2100.0,
                peak_rss_mb=400.0,
                vram_available=True,
                sample_count=2,
            )
        )

    created: list[FakeWorkflow] = []
    _rows, summary = run_gpu_cpu_benchmark(
        QUESTIONS,
        modes=(BenchmarkMode(name="gpu", num_gpu=999, requires_gpu=True),),
        workflow_factory=_build_factory(created),
        gpu_detector=lambda model: GpuEngagement(
            on_gpu=True, partial_offload=False, processor="100% GPU"
        ),
        sampler_factory=sampler_factory,
        unloader=lambda model: None,
        ps_vram_runner=lambda: GPU_PS_OUTPUT,
        global_vram_reader=lambda: 1500.0,
        compute_apps_runner=lambda: "",
        warmup_question="warmup",
        logger=lambda message: None,
    )

    gpu = summary["modes"]["gpu"]
    assert gpu["baseline_global_vram_mb"] == 1500.0
    assert gpu["peak_global_vram_mb"] == 2100.0
    assert gpu["vram_delta_mb"] == 600.0


def test_wddm_compute_apps_na_does_not_crash_and_is_not_primary():
    # WDDM reports per-process memory as "[N/A]"; the primary value must still
    # come from ollama ps, and compute_apps must not crash the run.
    wddm_output = "1234, ollama.exe, [N/A]\n"
    assert query_ollama_vram_mb(runner=lambda: wddm_output) == 0.0

    created: list[FakeWorkflow] = []
    _rows, summary = run_gpu_cpu_benchmark(
        QUESTIONS,
        modes=(BenchmarkMode(name="gpu", num_gpu=999, requires_gpu=True),),
        workflow_factory=_build_factory(created),
        gpu_detector=lambda model: GpuEngagement(
            on_gpu=True, partial_offload=False, processor="100% GPU"
        ),
        sampler_factory=_sampler_factory,
        unloader=lambda model: None,
        ps_vram_runner=lambda: GPU_PS_OUTPUT,
        global_vram_reader=lambda: 1000.0,
        compute_apps_runner=lambda: wddm_output,
        warmup_question="warmup",
        logger=lambda message: None,
    )

    gpu = summary["modes"]["gpu"]
    # Primary VRAM from ollama ps is non-zero; compute_apps is best-effort 0/N-A.
    assert gpu["vram_mb"] == GPU_SIZE_MB
    assert gpu["compute_apps_vram_mb"] == 0.0
    assert gpu["vram_mb"] != gpu["compute_apps_vram_mb"]


class _CapturingResponse:
    def raise_for_status(self) -> None:
        return None


def test_ollama_tool_unload_posts_keep_alive_zero(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_post(url, json, timeout):  # noqa: A002 - mirror requests.post signature
        captured["url"] = url
        captured["json"] = json
        return _CapturingResponse()

    monkeypatch.setattr(
        "local_agentic_analytics.tools.ollama_tool.requests.post", fake_post
    )
    tool = OllamaTool(base_url="http://localhost:11434", model="gemma2:2b")

    assert tool.unload() is True
    assert captured["url"].endswith("/api/generate")
    assert captured["json"] == {"model": "gemma2:2b", "keep_alive": 0}


# --- Repeated runs: aggregation (mean/sd/min/max) ------------------------------


def test_aggregate_stats_mean_sd_min_max():
    stats = aggregate_stats([2.0, 4.0, 6.0])

    assert stats["n"] == 3
    assert stats["mean"] == 4.0
    assert stats["sd"] == pytest.approx(2.0)  # sample stdev of [2, 4, 6]
    assert stats["min"] == 2.0
    assert stats["max"] == 6.0


def test_aggregate_stats_single_value_has_zero_sd():
    stats = aggregate_stats([5.0])

    assert stats == {"n": 1, "mean": 5.0, "sd": 0.0, "min": 5.0, "max": 5.0}


def test_aggregate_stats_ignores_none_and_handles_empty():
    assert aggregate_stats([None, 3.0])["n"] == 1
    assert aggregate_stats([None, None]) == {
        "n": 0,
        "mean": None,
        "sd": None,
        "min": None,
        "max": None,
    }


def _single_run_summary(latency: float, tps: float, vram: float) -> dict:
    return {
        "modes": {
            "gpu": {
                "label": "gpu",
                "avg_total_latency": latency,
                "avg_tokens_per_second": tps,
                "vram_mb": vram,
                "processor_split": "100% GPU",
                "vram_delta_mb": 1100.0,
            }
        }
    }


def test_aggregate_repeated_summaries_computes_per_mode_stats():
    summaries = [
        _single_run_summary(latency=2.0, tps=10.0, vram=1945.6),
        _single_run_summary(latency=4.0, tps=20.0, vram=1945.6),
    ]

    aggregated = aggregate_repeated_summaries(summaries)

    assert aggregated["repeat"] == 2
    gpu = aggregated["modes"]["gpu"]
    assert gpu["runs"] == 2
    assert gpu["latency_stats"]["mean"] == 3.0
    assert gpu["latency_stats"]["sd"] == pytest.approx(2.0**0.5)
    assert gpu["latency_stats"]["min"] == 2.0
    assert gpu["latency_stats"]["max"] == 4.0
    assert gpu["tokens_per_second_stats"]["mean"] == 15.0
    assert gpu["tokens_per_second_stats"]["sd"] == pytest.approx(50.0**0.5)
    # VRAM stays a single representative value (stable) plus min/max.
    assert gpu["vram_mb"] == 1945.6
    assert gpu["vram_mb_stats"]["min"] == 1945.6
    assert gpu["processor_split"] == "100% GPU"  # representative field preserved


def test_run_repeated_tags_run_index_and_aggregates():
    created: list[FakeWorkflow] = []
    rows, summary = run_gpu_cpu_benchmark_repeated(
        QUESTIONS,
        repeat=3,
        workflow_factory=_build_factory(created),
        gpu_detector=lambda model: GpuEngagement(
            on_gpu=True, partial_offload=False, processor="100% GPU"
        ),
        sampler_factory=_sampler_factory,
        unloader=lambda model: None,
        warmup_question="warmup",
        logger=lambda message: None,
        **_gpu_runners(),
    )

    # Per-run rows preserved with a run_index column for runs 1..3.
    assert sorted({row["run_index"] for row in rows}) == [1, 2, 3]
    assert len(rows) == 3 * 2 * len(QUESTIONS)  # repeats x modes x questions

    assert summary["repeat"] == 3
    gpu = summary["modes"]["gpu"]
    assert gpu["runs"] == 3
    assert gpu["latency_stats"]["n"] == 3
    assert gpu["latency_stats"]["mean"] == 1.0  # FakeWorkflow latency is fixed
    assert gpu["latency_stats"]["sd"] == 0.0
    assert gpu["tokens_per_second_stats"]["mean"] == 10.0
    assert gpu["tokens_per_second_stats"]["sd"] == 0.0
    assert gpu["vram_mb"] == GPU_SIZE_MB
    assert set(summary["modes"]) == {"gpu", "cpu"}


def test_run_repeated_rejects_invalid_repeat():
    with pytest.raises(ValueError):
        run_gpu_cpu_benchmark_repeated(QUESTIONS, repeat=0)
