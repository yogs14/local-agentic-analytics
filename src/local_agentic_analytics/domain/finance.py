"""Finance dataset profile and adapter (yfinance stock prices)."""

from __future__ import annotations

from local_agentic_analytics.domain.adapters import DomainAdapter
from local_agentic_analytics.domain.profile import (
    ColumnProfile,
    DatasetProfile,
    SemanticRule,
)


FINANCE_DATASET_PROFILE = DatasetProfile(
    dataset_id="sp500_analyst_ratings",
    display_name="Daily Stock Prices (yfinance) for NVDA, NFLX, TSLA, GOOGL",
    table_name="stock_prices",
    datetime_column="date",
    columns=(
        ColumnProfile(
            name="date",
            data_type="DATE",
            description="Trading date.",
        ),
        ColumnProfile(
            name="ticker",
            data_type="VARCHAR",
            description="Stock ticker symbol (NVDA, NFLX, TSLA, GOOGL).",
        ),
        ColumnProfile(
            name="open",
            data_type="DOUBLE",
            unit="USD",
            description="Daily opening price.",
        ),
        ColumnProfile(
            name="high",
            data_type="DOUBLE",
            unit="USD",
            description="Daily high price.",
        ),
        ColumnProfile(
            name="low",
            data_type="DOUBLE",
            unit="USD",
            description="Daily low price.",
        ),
        ColumnProfile(
            name="close",
            data_type="DOUBLE",
            unit="USD",
            description="Daily closing price.",
        ),
        ColumnProfile(
            name="volume",
            data_type="BIGINT",
            unit="shares",
            description="Daily traded volume in shares.",
        ),
    ),
    semantic_rules=(
        SemanticRule(
            rule_id="ticker_filter",
            description=(
                "Always filter by ticker when the question mentions a stock "
                "symbol such as NVDA, NFLX, TSLA, or GOOGL."
            ),
            sql_hint="WHERE ticker = 'TICKER'",
        ),
        SemanticRule(
            rule_id="average_close",
            description="If the user asks average close price, use a USD alias.",
            sql_hint="AVG(close) AS avg_close_usd",
        ),
        SemanticRule(
            rule_id="maximum_close",
            description="If the user asks the highest close price, aggregate with MAX.",
            sql_hint="MAX(close) AS max_close_usd",
        ),
        SemanticRule(
            rule_id="average_volume",
            description="If the user asks average trading volume, use a shares alias.",
            sql_hint="AVG(volume) AS avg_volume_shares",
        ),
        SemanticRule(
            rule_id="price_change",
            description=(
                "Day-over-day price change uses a window function ordered by date."
            ),
            sql_hint=(
                "close - LAG(close) OVER (ORDER BY date) AS price_change_usd"
            ),
        ),
        SemanticRule(
            rule_id="date_filter",
            description=(
                "For whole-day filters on the date column, cast to DATE and "
                "compare against DATE literals."
            ),
            sql_hint="CAST(date AS DATE) = DATE 'YYYY-MM-DD'",
        ),
        SemanticRule(
            rule_id="missing_value_count",
            description="Missing value counts must use DuckDB FILTER syntax.",
            sql_hint=(
                "COUNT(*) FILTER (WHERE <column> IS NULL) "
                "AS missing_<column>_count"
            ),
        ),
    ),
)


class FinanceDomainAdapter(DomainAdapter):
    def __init__(self, profile: DatasetProfile = FINANCE_DATASET_PROFILE):
        super().__init__(profile=profile)

    def format_sql_prompt_context(self) -> str:
        return super().format_sql_prompt_context() + "\n\n" + FINANCE_FEW_SHOT_EXAMPLES


FINANCE_FEW_SHOT_EXAMPLES = """Domain few-shot examples:

Question: Berapa rata-rata harga penutupan NVDA antara 2 Januari 2019 dan 31 Januari 2019?
SQL:
SELECT AVG(close) AS avg_close_usd
FROM stock_prices
WHERE ticker = 'NVDA'
  AND CAST(date AS DATE) BETWEEN DATE '2019-01-02' AND DATE '2019-01-31';

Question: Berapa harga penutupan tertinggi TSLA?
SQL:
SELECT MAX(close) AS max_close_usd
FROM stock_prices
WHERE ticker = 'TSLA';

Question: Bandingkan harga penutupan NVDA dan GOOGL pada tanggal 2019-06-03.
SQL:
SELECT ticker, close AS close_usd
FROM stock_prices
WHERE ticker IN ('NVDA', 'GOOGL')
  AND CAST(date AS DATE) = DATE '2019-06-03';

Question: Berapa jumlah hari perdagangan NFLX pada Januari 2019?
SQL:
SELECT COUNT(*) AS trading_days
FROM stock_prices
WHERE ticker = 'NFLX'
  AND date >= DATE '2019-01-01' AND date < DATE '2019-02-01';

Question: Berapa perubahan harga penutupan harian GOOGL?
SQL:
SELECT date, close - LAG(close) OVER (ORDER BY date) AS price_change_usd
FROM stock_prices
WHERE ticker = 'GOOGL'
ORDER BY date;"""


def get_finance_domain_adapter() -> FinanceDomainAdapter:
    return FinanceDomainAdapter()
