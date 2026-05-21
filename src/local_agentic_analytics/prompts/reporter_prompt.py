"""Prompt template for Indonesian analytics reporting."""

from __future__ import annotations

import json
from typing import Any


REPORTER_PROMPT_TEMPLATE = """Anda adalah reporter analitik data.

Tugas:
Jawab pertanyaan user dalam bahasa Indonesia yang jelas dan singkat berdasarkan SQL
yang dieksekusi dan hasil query.

Aturan:
- Gunakan hanya informasi yang ada pada hasil query.
- Jangan mengarang angka, tren, penyebab, atau kesimpulan di luar hasil query.
- Jika hasil query kosong, katakan bahwa tidak ada data yang ditemukan.
- Jika data tidak cukup untuk menyimpulkan anomali, katakan bahwa perlu pembanding historis.
- Jangan menjelaskan proses internal.
- Jawaban maksimal 2 paragraf pendek.

Pertanyaan user:
{question}

SQL yang dieksekusi:
{sql}

Hasil query:
{query_result}

Jawaban bahasa Indonesia:
"""


def build_reporter_prompt(question: str, sql: str, query_result: Any) -> str:
    """Build a prompt for concise Indonesian reporting."""
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    if not sql or not sql.strip():
        raise ValueError("sql must not be empty")

    formatted_result = format_query_result(query_result)
    if not formatted_result:
        raise ValueError("query_result must not be empty")

    return REPORTER_PROMPT_TEMPLATE.format(
        question=question.strip(),
        sql=sql.strip(),
        query_result=formatted_result,
    )


def format_query_result(query_result: Any) -> str:
    """Format compact query results for the reporter prompt."""
    if query_result is None:
        return ""

    if isinstance(query_result, str):
        return query_result.strip()

    try:
        return json.dumps(query_result, ensure_ascii=False, indent=2)
    except TypeError:
        return str(query_result).strip()
