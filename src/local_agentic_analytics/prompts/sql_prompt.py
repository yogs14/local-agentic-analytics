"""Prompt template for DuckDB SQL generation."""

from __future__ import annotations

from local_agentic_analytics.core.dataset_profile import (
    load_dataset_profile,
    profile_to_compact_sql_context,
)


ENERGY_FEW_SHOT_EXAMPLES = """Few-shot examples:

Example 1:
Q: Berapa rata-rata daya aktif pada tanggal 16 Desember 2006?
SQL:
SELECT AVG(Global_active_power) AS avg_global_active_power_kw
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2006-12-16';

Example 2:
Q: Berapa total energi kWh pada tanggal 16 Desember 2006 berdasarkan Global_active_power?
SQL:
SELECT SUM(Global_active_power) / 60.0 AS total_energy_kwh
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2006-12-16';

Example 3:
Q: Berapa jumlah missing value pada kolom Global_active_power?
SQL:
SELECT COUNT(*) FILTER (WHERE Global_active_power IS NULL) AS missing_global_active_power_count
FROM electric_power;"""


SQL_PROMPT_TEMPLATE = """You are a DuckDB SQL generator.

Task:
Generate one valid DuckDB SQL query that answers the user question.

Rules:
- Return SQL only.
- No markdown, no code fences, no explanation.
- Use only the exact table name and columns present in the schema.
- Use semantic rules from the compact DatasetProfile context when they apply.
- Alias names are output names only.
- Never SELECT an alias as a source column.
- Aliases from DatasetProfile are not input columns; use them only after aggregate/calculation expressions.
- Do not invent generic table names such as data, dataset, or table.
- Do not rename columns. For example, do not use timestamp if the schema has datetime.
- Do not add catalog or schema prefixes unless they are present in the schema.
- For date filters on a TIMESTAMP column, prefer CAST(column AS DATE) = DATE 'YYYY-MM-DD'.
- Never compare a TIMESTAMP column directly to a DATE when the user asks about a whole day.
- Do not use Date or Time columns for filtering if datetime is available.
- If the user asks total energi, energi kWh, total kWh, or konsumsi energi, use:
  SUM(Global_active_power) / 60.0 AS total_energy_kwh
- If the user asks rata-rata daya aktif, use:
  AVG(Global_active_power) AS avg_global_active_power_kw
- For rata-rata daya aktif with a date filter, use:
  SELECT AVG(Global_active_power) AS avg_global_active_power_kw
  FROM electric_power
  WHERE CAST(datetime AS DATE) = DATE 'YYYY-MM-DD';
- Never write:
  SELECT avg_global_active_power_kw FROM electric_power
- If the user asks jumlah missing value, use:
  COUNT(*) FILTER (WHERE column IS NULL) AS missing_<column>_count
- If the user asks record count or jumlah record, use:
  COUNT(*) AS record_count
- Never use COUNT(DISTINCT CASE WHEN ... THEN 1 END) for missing value counts.
- Always use clear aliases with units: _kw, _kwh, _v, _a, or _count.

- If the user asks for maximum active power value, use:
  MAX(Global_active_power) AS max_global_active_power_kw

Dataset profile and semantic rules:
{dataset_profile_context}

Date filter example:
If the schema has datetime: TIMESTAMP and the user asks for 16 December 2006,
use WHERE CAST(datetime AS DATE) = DATE '2006-12-16'.

Negative alias example:
Wrong:
SELECT avg_global_active_power_kw FROM electric_power

Correct:
SELECT AVG(Global_active_power) AS avg_global_active_power_kw
FROM electric_power

{few_shot_examples}

Schema:
{schema}

User question:
{question}

SQL:
"""


def build_sql_prompt(
    question: str,
    schema: str,
    dataset_profile_context: str | None = None,
    domain_context: str | None = None,
) -> str:
    """Build a prompt for SQL-only generation."""
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    if not schema or not schema.strip():
        raise ValueError("schema must not be empty")

    few_shot_examples = ""
    if dataset_profile_context is None:
        dataset_profile_context = domain_context
    if dataset_profile_context is None:
        dataset_profile_context = profile_to_compact_sql_context(
            load_dataset_profile("energy")
        )
        few_shot_examples = ENERGY_FEW_SHOT_EXAMPLES

    return SQL_PROMPT_TEMPLATE.format(
        question=question.strip(),
        schema=schema.strip(),
        dataset_profile_context=dataset_profile_context.strip(),
        few_shot_examples=few_shot_examples.strip(),
    )
