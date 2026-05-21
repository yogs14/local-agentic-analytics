"""Prompt template for DuckDB SQL generation."""

from __future__ import annotations


SQL_PROMPT_TEMPLATE = """You are a DuckDB SQL generator.

Task:
Generate one valid DuckDB SQL query that answers the user question.

Rules:
- Return SQL only.
- Do not use markdown.
- Do not use code fences.
- Do not explain the query.
- Use only the exact table name and columns present in the schema.
- Do not invent generic table names such as data, dataset, or table.
- Do not rename columns. For example, do not use timestamp if the schema has datetime.
- Do not add catalog or schema prefixes unless they are present in the schema.
- If a datetime column is available, use DuckDB datetime functions when needed.
- For date filters on a TIMESTAMP column, prefer CAST(column AS DATE) = DATE 'YYYY-MM-DD'.
- Never compare a TIMESTAMP column directly to a DATE when the user asks about a whole day.

Date filter example:
If the schema has datetime: TIMESTAMP and the user asks for 16 December 2006,
use WHERE CAST(datetime AS DATE) = DATE '2006-12-16'.

Schema:
{schema}

User question:
{question}

SQL:
"""


def build_sql_prompt(question: str, schema: str) -> str:
    """Build a prompt for SQL-only generation."""
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    if not schema or not schema.strip():
        raise ValueError("schema must not be empty")

    return SQL_PROMPT_TEMPLATE.format(
        question=question.strip(),
        schema=schema.strip(),
    )
