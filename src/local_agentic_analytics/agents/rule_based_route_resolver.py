"""Deterministic retrieval-route resolver (the rule-based half of the planner).

Mirrors the rule-based-vs-LLM pattern already used for SQL: this pure, testable
resolver runs first and only defers to the LLM planner when it is not confident.
It never raises and never reaches the model.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from local_agentic_analytics.agents.route_types import RouteDecision


# News / unstructured-retrieval signals (Bahasa Indonesia, case-insensitive).
RAG_SIGNALS = ("berita", "headline", "sentimen", "analis", "publisher", "kabar")
# Price / structured-movement signals.
PRICE_SIGNALS = ("harga", "pergerakan", "tren", "performa", "close")
# Fusion connectors that link a price summary to its news.
HYBRID_CONNECTORS = (
    "kaitkan",
    "beserta",
    "sekaligus",
    "dan apa berita",
    "relevan",
)


@dataclass(frozen=True)
class RouteResolution:
    """Outcome of the deterministic route resolver.

    ``confident`` is True only when an explicit non-SQL signal was found. When it
    is False the route defaults to STRUCTURED_SQL but the caller may still ask the
    LLM planner for a better-informed decision.
    """

    route: RouteDecision
    reasoning: str
    confident: bool


class RuleBasedRouteResolver:
    """Resolve a retrieval route from cheap deterministic keyword heuristics."""

    def resolve(self, question: str, domain: str = "finance") -> RouteResolution:
        normalized_domain = (domain or "").strip().lower()
        if normalized_domain == "energy":
            return RouteResolution(
                route=RouteDecision.STRUCTURED_SQL,
                reasoning="Energy domain selalu STRUCTURED_SQL.",
                confident=True,
            )

        if not question or not question.strip():
            return RouteResolution(
                route=RouteDecision.STRUCTURED_SQL,
                reasoning="Pertanyaan kosong; default STRUCTURED_SQL.",
                confident=False,
            )

        text = _normalize_text(question)
        rag_hits = _matched(text, RAG_SIGNALS)
        price_hits = _matched(text, PRICE_SIGNALS)
        connector_hits = _matched(text, HYBRID_CONNECTORS)

        if rag_hits and (price_hits or connector_hits):
            # Both a news signal and a price/connector signal: the two sources must
            # be fused. Hybrid wins over pure RAG when both are present.
            return RouteResolution(
                route=RouteDecision.HYBRID,
                reasoning=(
                    "Sinyal berita + harga/konektor: "
                    f"{_join(rag_hits + price_hits + connector_hits)}."
                ),
                confident=True,
            )

        if rag_hits:
            return RouteResolution(
                route=RouteDecision.RAG_NEWS,
                reasoning=f"Sinyal berita: {_join(rag_hits)}.",
                confident=True,
            )

        return RouteResolution(
            route=RouteDecision.STRUCTURED_SQL,
            reasoning="Tidak ada sinyal berita; default STRUCTURED_SQL.",
            confident=False,
        )


def _matched(text: str, signals: tuple[str, ...]) -> list[str]:
    return [signal for signal in signals if signal in text]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _join(signals: list[str]) -> str:
    seen: list[str] = []
    for signal in signals:
        if signal not in seen:
            seen.append(signal)
    return ", ".join(seen)
