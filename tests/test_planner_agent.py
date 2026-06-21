from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.planner_agent import (
    PlannerAgent,
    parse_route_label,
)
from local_agentic_analytics.agents.route_types import RouteDecision


class FakeOllamaTool:
    def __init__(self, response: str, metrics=None):
        self.response = response
        self.metrics = metrics or {}
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

    def get_last_metrics(self):
        return dict(self.metrics)


def test_planner_parses_clean_label():
    tool = FakeOllamaTool("RAG_NEWS")
    agent = PlannerAgent(ollama_tool=tool)

    route = agent.plan_route("Bagaimana sentimen berita terbaru tentang TSLA?")

    assert route == RouteDecision.RAG_NEWS
    assert tool.calls[0]["temperature"] == 0.0
    assert tool.calls[0]["max_tokens"] == 16


def test_planner_parses_label_embedded_in_prose():
    tool = FakeOllamaTool("Menurut saya rutenya adalah HYBRID karena ada harga dan berita.")
    agent = PlannerAgent(ollama_tool=tool)

    route = agent.plan_route("Ringkas harga NVDA dan kaitkan beritanya.")

    assert route == RouteDecision.HYBRID


def test_planner_falls_back_to_sql_on_garbage_output():
    tool = FakeOllamaTool("entah apa ini bukan label valid 123")
    agent = PlannerAgent(ollama_tool=tool)

    route = agent.plan_route("Berapa rata-rata harga penutupan NVDA?")

    assert route == RouteDecision.STRUCTURED_SQL


def test_planner_falls_back_to_sql_on_empty_output():
    tool = FakeOllamaTool("")
    agent = PlannerAgent(ollama_tool=tool)

    route = agent.plan_route("Berapa rata-rata harga penutupan NVDA?")

    assert route == RouteDecision.STRUCTURED_SQL


def test_planner_exposes_last_metrics():
    tool = FakeOllamaTool(
        "STRUCTURED_SQL",
        metrics={"total_duration": 1.5, "eval_count": 3},
    )
    agent = PlannerAgent(ollama_tool=tool)

    agent.plan_route("Berapa harga penutupan tertinggi TSLA?")

    metrics = agent.get_last_metrics()
    assert metrics["total_duration"] == 1.5
    assert metrics["eval_count"] == 3


def test_parse_route_label_is_case_insensitive():
    assert parse_route_label("rag_news") == RouteDecision.RAG_NEWS
    assert parse_route_label("  Hybrid  ") == RouteDecision.HYBRID
    assert parse_route_label(None) == RouteDecision.STRUCTURED_SQL
