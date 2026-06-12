from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.domain.energy import get_energy_domain_adapter
from local_agentic_analytics.domain.profile import (
    ColumnProfile,
    DatasetProfile,
    SemanticRule,
)


def test_energy_domain_adapter_exposes_profile_and_semantic_rules():
    adapter = get_energy_domain_adapter()
    context = adapter.format_sql_prompt_context()

    assert adapter.table_name == "electric_power"
    assert "Individual Household Electric Power Consumption" in context
    assert "Global_active_power" in context
    assert "SUM(Global_active_power) / 60.0 AS total_energy_kwh" in context
    assert "CAST(datetime AS DATE) = DATE 'YYYY-MM-DD'" in context


def test_dataset_profile_formats_columns_and_rules_for_prompt():
    profile = DatasetProfile(
        dataset_id="sales",
        display_name="Sales",
        table_name="sales",
        columns=(
            ColumnProfile(
                name="amount",
                data_type="DOUBLE",
                unit="USD",
                description="Transaction amount.",
            ),
        ),
        semantic_rules=(
            SemanticRule(
                rule_id="total_sales",
                description="Use SUM for total sales.",
                sql_hint="SUM(amount) AS total_amount_usd",
            ),
        ),
    )

    assert "amount: DOUBLE; unit=USD; Transaction amount." in (
        profile.format_columns_for_prompt()
    )
    assert "SUM(amount) AS total_amount_usd" in profile.format_rules_for_prompt()
