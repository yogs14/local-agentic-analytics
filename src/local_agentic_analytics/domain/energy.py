"""Energy dataset profile and adapter."""

from __future__ import annotations

from local_agentic_analytics.domain.adapters import DomainAdapter
from local_agentic_analytics.domain.profile import (
    ColumnProfile,
    DatasetProfile,
    SemanticRule,
)


ENERGY_DATASET_PROFILE = DatasetProfile(
    dataset_id="household_power_consumption",
    display_name="Individual Household Electric Power Consumption",
    table_name="electric_power",
    datetime_column="datetime",
    columns=(
        ColumnProfile(
            name="datetime",
            data_type="TIMESTAMP",
            description="Combined Date and Time column for minute-level readings.",
        ),
        ColumnProfile(
            name="Global_active_power",
            data_type="DOUBLE",
            unit="kW",
            description="Global active power.",
        ),
        ColumnProfile(
            name="Global_reactive_power",
            data_type="DOUBLE",
            unit="kW",
            description="Global reactive power.",
        ),
        ColumnProfile(
            name="Voltage",
            data_type="DOUBLE",
            unit="Volt",
            description="Average voltage.",
        ),
        ColumnProfile(
            name="Global_intensity",
            data_type="DOUBLE",
            unit="Ampere",
            description="Global current intensity.",
        ),
        ColumnProfile(
            name="Sub_metering_1",
            data_type="DOUBLE",
            unit="Wh",
            description="Sub-metering channel 1.",
        ),
        ColumnProfile(
            name="Sub_metering_2",
            data_type="DOUBLE",
            unit="Wh",
            description="Sub-metering channel 2.",
        ),
        ColumnProfile(
            name="Sub_metering_3",
            data_type="DOUBLE",
            unit="Wh",
            description="Sub-metering channel 3.",
        ),
    ),
    semantic_rules=(
        SemanticRule(
            rule_id="total_energy_kwh",
            description=(
                "If the user asks total energi, energi kWh, total kWh, or "
                "konsumsi energi, compute energy from minute-level kW readings."
            ),
            sql_hint="SUM(Global_active_power) / 60.0 AS total_energy_kwh",
        ),
        SemanticRule(
            rule_id="average_active_power",
            description="If the user asks average active power, use a kW alias.",
            sql_hint="AVG(Global_active_power) AS avg_global_active_power_kw",
        ),
        SemanticRule(
            rule_id="missing_value_count",
            description="Missing value counts must use DuckDB FILTER syntax.",
            sql_hint=(
                "COUNT(*) FILTER (WHERE <column> IS NULL) "
                "AS missing_<column>_count"
            ),
        ),
        SemanticRule(
            rule_id="missing_value_anti_pattern",
            description=(
                "Never use COUNT(DISTINCT CASE WHEN ... THEN 1 END) for "
                "missing values."
            ),
        ),
        SemanticRule(
            rule_id="record_count",
            description="If the user asks record count or jumlah record, count rows.",
            sql_hint="COUNT(*) AS record_count",
        ),
        SemanticRule(
            rule_id="date_filter",
            description=(
                "For whole-day filters on datetime, always cast datetime to DATE "
                "and never compare TIMESTAMP directly to DATE."
            ),
            sql_hint="CAST(datetime AS DATE) = DATE 'YYYY-MM-DD'",
        ),
        SemanticRule(
            rule_id="voltage_average",
            description="If the user asks average voltage, use Volt unit alias.",
            sql_hint="AVG(Voltage) AS avg_voltage_v",
        ),
        SemanticRule(
            rule_id="current_average",
            description="If the user asks average current, use Ampere unit alias.",
            sql_hint="AVG(Global_intensity) AS avg_global_intensity_a",
        ),
        SemanticRule(
            rule_id="maximum_active_power",
            description=(
                "If the user asks maximum active power value, aggregate with MAX. "
                "If the user asks when it happened, return datetime plus the value."
            ),
            sql_hint=(
                "MAX(Global_active_power) AS max_global_active_power_kw; "
                "datetime, Global_active_power AS max_global_active_power_kw"
            ),
        ),
    ),
)


class EnergyDomainAdapter(DomainAdapter):
    def __init__(self, profile: DatasetProfile = ENERGY_DATASET_PROFILE):
        super().__init__(profile=profile)

    def format_sql_prompt_context(self) -> str:
        return super().format_sql_prompt_context() + "\n\n" + ENERGY_FEW_SHOT_EXAMPLES


ENERGY_FEW_SHOT_EXAMPLES = """Domain few-shot examples:

Question: Berapa rata-rata daya aktif pada tanggal 16 Desember 2006?
SQL:
SELECT AVG(Global_active_power) AS avg_global_active_power_kw
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2006-12-16';

Question: Berapa total energi kWh pada tanggal 16 Desember 2006 berdasarkan Global_active_power?
SQL:
SELECT SUM(Global_active_power) / 60.0 AS total_energy_kwh
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2006-12-16';

Question: Berapa jumlah missing value pada kolom Global_active_power?
SQL:
SELECT COUNT(*) FILTER (WHERE Global_active_power IS NULL) AS missing_global_active_power_count
FROM electric_power;

Question: Berapa jumlah record pada tanggal 16 Desember 2006?
SQL:
SELECT COUNT(*) AS record_count
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2006-12-16';

Question: Berapa tegangan rata-rata pada tanggal 16 Desember 2006?
SQL:
SELECT AVG(Voltage) AS avg_voltage_v
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2006-12-16';

Question: Pada jam berapa daya aktif tertinggi terjadi pada tanggal 16 Desember 2006?
SQL:
SELECT datetime, Global_active_power AS max_global_active_power_kw
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2006-12-16'
ORDER BY Global_active_power DESC
LIMIT 1;"""


def get_energy_domain_adapter() -> EnergyDomainAdapter:
    return EnergyDomainAdapter()
