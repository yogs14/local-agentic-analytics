"""Prompt template for DuckDB SQL repair."""

from __future__ import annotations


REPAIR_PROMPT_TEMPLATE = """You are a DuckDB SQL repair assistant.

Task:
Fix the failed DuckDB SQL query using the DuckDB error message and schema.

Rules:
- Return SQL only.
- Do not use markdown.
- Do not use code fences.
- Do not explain the fix.
- Use only the exact table name and columns present in the schema.
- Do not invent generic table names such as data, dataset, or table.
- Do not rename columns. For example, do not use timestamp if the schema has datetime.
- Do not add catalog or schema prefixes unless they are present in the schema.
- Never compare a TIMESTAMP column directly to a DATE when the user asks about a whole day.
- For date filters on a TIMESTAMP column, use CAST(column AS DATE) = DATE 'YYYY-MM-DD'.
- Preserve the original analytical intent when possible.
- Produce one corrected DuckDB SQL query.

Schema:
{schema}

Failed SQL:
{failed_sql}

DuckDB error:
{error_message}

Corrected SQL:
"""


def build_repair_prompt(failed_sql: str, error_message: str, schema: str) -> str:
    """Build a prompt for SQL-only repair."""
    if not failed_sql or not failed_sql.strip():
        raise ValueError("failed_sql must not be empty")
    if not error_message or not error_message.strip():
        raise ValueError("error_message must not be empty")
    if not schema or not schema.strip():
        raise ValueError("schema must not be empty")

    return REPAIR_PROMPT_TEMPLATE.format(
        failed_sql=failed_sql.strip(),
        error_message=error_message.strip(),
        schema=schema.strip(),
    )
