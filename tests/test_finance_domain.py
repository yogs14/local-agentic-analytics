from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core.dataset_profile import load_dataset_profile
from local_agentic_analytics.domain.finance import (
    FinanceDomainAdapter,
    get_finance_domain_adapter,
)
from local_agentic_analytics.domain.registry import get_domain_adapter


def test_finance_profile_yaml_loads_hybrid_schema():
    profile = load_dataset_profile("finance")

    assert profile.name == "sp500_analyst_ratings"
    assert profile.domain == "finance"
    assert profile.table_name == "stock_prices"
    assert profile.datetime_column == "date"
    assert set(profile.columns) == {
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    assert profile.columns["close"].unit == "USD"


def test_finance_adapter_context_contains_key_rules():
    adapter = get_finance_domain_adapter()
    context = adapter.format_sql_prompt_context()

    assert adapter.table_name == "stock_prices"
    assert "AVG(close) AS avg_close_usd" in context
    assert "MAX(close) AS max_close_usd" in context
    assert "AVG(volume) AS avg_volume_shares" in context
    assert "WHERE ticker = 'TICKER'" in context
    assert "LAG(close) OVER (ORDER BY date)" in context
    # Few-shot examples must be present so they reach the SQL prompt.
    assert "Domain few-shot examples:" in context
    assert "FROM stock_prices" in context


def test_registry_returns_finance_adapter_only_for_finance():
    assert isinstance(get_domain_adapter("finance"), FinanceDomainAdapter)
    assert get_domain_adapter("energy") is None
    assert get_domain_adapter("") is None
    assert get_domain_adapter(None) is None
