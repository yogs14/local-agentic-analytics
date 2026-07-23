from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.insight_agent import (
    InsightAgent,
    sanitize_narrative,
)
from local_agentic_analytics.prompts.insight_prompt import (
    build_energy_finetune_prompt,
    build_insight_prompt,
)


class FakeOllamaTool:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 512) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self.response


def test_build_insight_prompt_contains_metadata_stats_and_rules():
    prompt = build_insight_prompt(
        {
            "chart_id": "daily_active_power_trend",
            "chart_title": "Tren Rata-rata Daya Aktif Harian",
            "chart_path": "reports/figures/daily_active_power_trend.png",
            "stats": {
                "mean_daily_avg_kw": 1.234,
                "unit": "kW",
            },
        }
    )

    assert "daily_active_power_trend" in prompt
    assert "Tren Rata-rata Daya Aktif Harian" in prompt
    assert '"mean_daily_avg_kw": 1.234' in prompt
    assert "Jangan mengarang angka" in prompt
    assert "pembanding historis" in prompt
    assert "Global_active_power = kW" in prompt


def test_insight_agent_returns_response_from_ollama():
    fake_tool = FakeOllamaTool(
        "Grafik menunjukkan rata-rata daya aktif harian sebesar 1,23 kW."
    )
    agent = InsightAgent(ollama_tool=fake_tool, temperature=0.1, max_tokens=256)

    insight = agent.generate_insight(
        {
            "chart_id": "daily_active_power_trend",
            "chart_title": "Tren Rata-rata Daya Aktif Harian",
            "chart_path": "reports/figures/daily_active_power_trend.png",
            "stats": {
                "mean_daily_avg_kw": 1.23,
                "unit": "kW",
            },
        }
    )

    assert insight == "Grafik menunjukkan rata-rata daya aktif harian sebesar 1,23 kW."
    assert fake_tool.calls[0]["temperature"] == 0.1
    assert fake_tool.calls[0]["max_tokens"] == 256


def test_insight_prompt_rejects_empty_stats():
    with pytest.raises(ValueError, match="stats must be a non-empty dict"):
        build_insight_prompt(
            {
                "chart_id": "daily_active_power_trend",
                "chart_title": "Tren Rata-rata Daya Aktif Harian",
                "chart_path": "reports/figures/daily_active_power_trend.png",
                "stats": {},
            }
        )


def test_energy_finetune_prompt_renders_flat_block():
    prompt = build_energy_finetune_prompt(
        {
            "chart_id": "voltage_distribution",
            "chart_title": "Distribusi Voltage",
            "chart_path": "reports/figures/voltage_distribution.png",
            "stats": {
                "record_count": 2_049_280,
                "avg_voltage_v": 240.1234,
                "min_voltage_v": 228.9,
                "unit": "Volt",
            },
        }
    )

    lines = prompt.splitlines()
    # Leading chart_id and flat "key: value" lines, no JSON braces or rules.
    assert lines[0] == "chart_id: voltage_distribution"
    assert "record_count: 2049280" in lines
    # Indonesian comma decimals, matching the fine-tune training format.
    assert "avg_voltage_v: 240,1234" in lines
    assert "unit: Volt" in lines
    assert "{" not in prompt
    assert "Jangan mengarang angka" not in prompt


def test_energy_finetune_prompt_flattens_correlation_stats():
    prompt = build_energy_finetune_prompt(
        {
            "chart_id": "correlation_heatmap",
            "stats": {
                "numeric_columns": ["Global_active_power", "Voltage"],
                "pair_count": 21,
                "correlations": {"Global_active_power vs Voltage": 0.12},
                "strongest_positive": {
                    "pair": "Global_active_power vs Global_intensity",
                    "correlation": 0.998,
                },
                "strongest_negative": {
                    "pair": "Voltage vs Global_intensity",
                    "correlation": -0.41,
                },
                "unit_note": "tanpa satuan",
            },
        }
    )

    assert "strongest_positive_pair: Global_active_power vs Global_intensity" in prompt
    assert "strongest_positive_r: 0,998" in prompt
    assert "strongest_negative_r: -0,41" in prompt
    # The full correlation matrix dict must not leak as a Python repr.
    assert "correlations:" not in prompt
    assert "{" not in prompt


def test_energy_finetune_prompt_requires_chart_id_and_stats():
    with pytest.raises(ValueError, match="chart_id"):
        build_energy_finetune_prompt({"chart_id": "", "stats": {"unit": "Volt"}})

    with pytest.raises(ValueError, match="stats must be a non-empty dict"):
        build_energy_finetune_prompt({"chart_id": "voltage_distribution", "stats": {}})


def test_sanitize_narrative_strips_markdown_but_keeps_identifiers_and_numbers():
    raw = (
        "## Pola Konsumsi\n\nRata-rata **240,8399 V** pada **Sub_metering_3** "
        "dengan Global_active_power 1,092 kW."
    )

    cleaned = sanitize_narrative(raw)

    assert "##" not in cleaned
    assert "**" not in cleaned
    assert "Sub_metering_3" in cleaned  # underscores preserved
    assert "Global_active_power" in cleaned
    assert "240,8399 V" in cleaned
    assert "1,092 kW" in cleaned


def test_insight_agent_sanitizes_markdown_from_model_output():
    fake_tool = FakeOllamaTool("## Judul\n\nTegangan rata-rata **240,12 V** stabil.")
    agent = InsightAgent(ollama_tool=fake_tool)

    insight = agent.generate_insight(
        {
            "chart_id": "voltage_distribution",
            "chart_title": "Distribusi Voltage",
            "chart_path": "reports/figures/voltage_distribution.png",
            "stats": {"avg_voltage_v": 240.12, "unit": "Volt"},
        }
    )

    assert "##" not in insight
    assert "**" not in insight
    assert "240,12 V" in insight


def test_for_energy_finetune_uses_flat_block_and_training_temperature():
    fake_tool = FakeOllamaTool("Narasi tegangan stabil pada 240,12 V.")
    agent = InsightAgent.for_energy_finetune(ollama_tool=fake_tool)

    assert agent.temperature == 0.4
    assert agent.prompt_builder is build_energy_finetune_prompt

    agent.generate_insight(
        {
            "chart_id": "voltage_distribution",
            "chart_title": "Distribusi Voltage",
            "chart_path": "reports/figures/voltage_distribution.png",
            "stats": {"avg_voltage_v": 240.12, "unit": "Volt"},
        }
    )

    sent_prompt = fake_tool.calls[0]["prompt"]
    assert sent_prompt.startswith("chart_id: voltage_distribution")
    assert fake_tool.calls[0]["temperature"] == 0.4


def test_insight_agent_raises_for_empty_model_response():
    fake_tool = FakeOllamaTool("  ")
    agent = InsightAgent(ollama_tool=fake_tool)

    with pytest.raises(RuntimeError, match="empty response"):
        agent.generate_insight(
            {
                "chart_id": "power_distribution",
                "chart_title": "Distribusi Global Active Power",
                "chart_path": "reports/figures/power_distribution.png",
                "stats": {
                    "avg_global_active_power_kw": 1.23,
                    "unit": "kW",
                },
            }
        )
