from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.insight_agent import InsightAgent
from local_agentic_analytics.prompts.insight_prompt import build_insight_prompt


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
