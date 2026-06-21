"""Domain adapter registry.

Maps a domain name to its optional DomainAdapter. The adapter supplies
few-shot examples that are injected into the SQL prompt. Energy intentionally
returns ``None`` to keep its existing prompt path byte-for-byte unchanged;
finance returns the FinanceDomainAdapter so its few-shots reach the SQL agent.
"""

from __future__ import annotations

from local_agentic_analytics.domain.adapters import DomainAdapter


def get_domain_adapter(domain: str | None) -> DomainAdapter | None:
    """Return the DomainAdapter for ``domain`` or ``None`` if unwired."""
    normalized = (domain or "").strip().lower()
    if normalized == "finance":
        from local_agentic_analytics.domain.finance import (
            get_finance_domain_adapter,
        )

        return get_finance_domain_adapter()

    return None
