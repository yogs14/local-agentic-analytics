from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.route_types import RouteDecision
from local_agentic_analytics.agents.rule_based_route_resolver import (
    RuleBasedRouteResolver,
)


def _resolver() -> RuleBasedRouteResolver:
    return RuleBasedRouteResolver()


def test_energy_domain_always_structured_sql_and_confident():
    resolution = _resolver().resolve(
        "Bagaimana sentimen berita terbaru tentang konsumsi listrik?",
        domain="energy",
    )

    assert resolution.route == RouteDecision.STRUCTURED_SQL
    assert resolution.confident is True


def test_pure_news_question_routes_to_rag():
    resolution = _resolver().resolve(
        "Bagaimana sentimen berita terbaru tentang TSLA?",
        domain="finance",
    )

    assert resolution.route == RouteDecision.RAG_NEWS
    assert resolution.confident is True


def test_publisher_question_routes_to_rag():
    resolution = _resolver().resolve(
        "Publisher mana yang paling sering memberitakan NFLX?",
        domain="finance",
    )

    assert resolution.route == RouteDecision.RAG_NEWS


def test_price_plus_news_with_connector_routes_to_hybrid():
    resolution = _resolver().resolve(
        "Ringkas pergerakan harga NVDA pada Juni 2019 dan kaitkan dengan beritanya.",
        domain="finance",
    )

    assert resolution.route == RouteDecision.HYBRID
    assert resolution.confident is True


def test_hybrid_wins_over_pure_rag_when_price_signal_present():
    resolution = _resolver().resolve(
        "Bagaimana performa harga TSLA pada awal 2020 dan apa berita pendukungnya?",
        domain="finance",
    )

    assert resolution.route == RouteDecision.HYBRID


def test_plain_aggregation_defaults_to_sql_without_confidence():
    resolution = _resolver().resolve(
        "Berapa rata-rata harga penutupan NVDA antara 2 Januari 2019 dan 31 Januari 2019?",
        domain="finance",
    )

    assert resolution.route == RouteDecision.STRUCTURED_SQL
    # No news signal: not confident, so the LLM planner may still be consulted.
    assert resolution.confident is False


def test_empty_question_defaults_to_sql_without_confidence():
    resolution = _resolver().resolve("   ", domain="finance")

    assert resolution.route == RouteDecision.STRUCTURED_SQL
    assert resolution.confident is False
