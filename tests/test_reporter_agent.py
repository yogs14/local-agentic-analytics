from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.reporter_agent import ReporterAgent
from local_agentic_analytics.prompts.reporter_prompt import (
    build_reporter_prompt,
    format_query_result,
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


def test_build_reporter_prompt_contains_question_sql_result_and_rules():
    prompt = build_reporter_prompt(
        question="Berapa rata-rata konsumsi listrik?",
        sql="SELECT AVG(Global_active_power) AS avg_power FROM electric_power",
        query_result=[{"avg_power": 1.23}],
    )

    assert "bahasa Indonesia" in prompt
    assert "Gunakan hanya informasi" in prompt
    assert "perlu pembanding historis" in prompt
    assert "Bulatkan angka desimal maksimal 2 atau 3 digit" in prompt
    # Units are tied to alias suffixes and inventing units is forbidden, so
    # arbitrary datasets without a unit do not get a hallucinated kW/MW.
    assert "_kw" in prompt and "_usd" in prompt
    assert "Jangan pernah mengarang satuan" in prompt
    assert "1-3 kalimat" in prompt
    assert "Jika hasil query hanya satu nilai numerik" in prompt
    assert "Jangan menambahkan insight" in prompt
    assert "AVG(Global_active_power)" in prompt
    assert '"avg_power": 1.23' in prompt


def test_format_query_result_accepts_compact_table_string():
    table = "avg_power\n1.23"

    assert format_query_result(table) == table


def test_reporter_agent_returns_indonesian_answer_from_ollama_response():
    fake_tool = FakeOllamaTool(
        "Rata-rata konsumsi listrik pada hasil query adalah 1,23 kW."
    )
    agent = ReporterAgent(ollama_tool=fake_tool, temperature=0.1, max_tokens=256)

    answer = agent.generate_answer(
        question="Berapa rata-rata konsumsi listrik?",
        sql="SELECT AVG(Global_active_power) AS avg_power FROM electric_power",
        query_result=[{"avg_power": 1.23}],
    )

    assert answer == "Rata-rata konsumsi listrik pada hasil query adalah 1,23 kW."
    assert fake_tool.calls[0]["temperature"] == 0.1
    assert fake_tool.calls[0]["max_tokens"] == 256


def test_reporter_agent_default_max_tokens_stays_concise():
    fake_tool = FakeOllamaTool("Jawaban ringkas berdasarkan hasil query.")
    agent = ReporterAgent(ollama_tool=fake_tool)

    answer = agent.generate_answer(
        question="Berapa rata-rata konsumsi listrik?",
        sql="SELECT AVG(Global_active_power) AS avg_power FROM electric_power",
        query_result=[{"avg_power": 1.23}],
    )

    assert answer == "Jawaban ringkas berdasarkan hasil query."
    assert fake_tool.calls[0]["max_tokens"] == 192


def test_reporter_agent_rejects_empty_query_result():
    fake_tool = FakeOllamaTool("Tidak ada data.")
    agent = ReporterAgent(ollama_tool=fake_tool)

    with pytest.raises(ValueError, match="query_result must not be empty"):
        agent.generate_answer(
            question="Ada anomali?",
            sql="SELECT * FROM electric_power LIMIT 0",
            query_result="",
        )


def test_reporter_agent_raises_for_empty_model_response():
    fake_tool = FakeOllamaTool("  ")
    agent = ReporterAgent(ollama_tool=fake_tool)

    with pytest.raises(RuntimeError, match="empty response"):
        agent.generate_answer(
            question="Berapa total konsumsi?",
            sql="SELECT SUM(Global_active_power) FROM electric_power",
            query_result=[{"sum": 100.0}],
        )
