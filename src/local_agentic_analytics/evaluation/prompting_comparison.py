"""Prompting-strategy comparison for text-to-SQL (Fase 4.1).

Compares three prompting strategies on the SAME model, gold set, temperature
(0.0), and schema context, as system-level alternatives to the existing
scaffolded pipeline:

- ``zero_shot``        - bare instruction + schema + question.
- ``few_shot_static``  - the same instruction plus a fixed set of exemplars
                         (identical for every question).
- ``decomposed``       - DIN-SQL-style three stages: schema linking -> draft
                         SQL -> self-refine. Three LLM calls per question,
                         no execution feedback.

SQL is extracted RAW (``extract_raw_model_sql``) — deliberately no rule-based
resolver, domain normalization, guard, or repair, because the strategies are
the system variable under test. Results reuse the exact comparison metrics of
the eval modules (legacy numeric match + row-set ``result_match_full``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from local_agentic_analytics.agents.sql_cleaning import extract_raw_model_sql
from local_agentic_analytics.core.dataset_profile import (
    load_dataset_profile,
    profile_to_compact_sql_context,
)
from local_agentic_analytics.evaluation.result_comparison import (
    compare_result_sets,
)
from local_agentic_analytics.evaluation.sql_gold_eval import (
    SqlExecutionResult,
    compare_numeric_results,
    execute_sql,
    load_gold_sql,
)


STRATEGY_ZERO_SHOT = "zero_shot"
STRATEGY_FEW_SHOT = "few_shot_static"
STRATEGY_DECOMPOSED = "decomposed"
STRATEGIES = (STRATEGY_ZERO_SHOT, STRATEGY_FEW_SHOT, STRATEGY_DECOMPOSED)

PROMPTING_COLUMNS = (
    "config",
    "question_id",
    "question",
    "agent_sql",
    "gold_sql",
    "execution_success",
    "agent_success",
    "gold_success",
    "numeric_match",
    "absolute_error",
    "relative_error",
    "error_message",
    "result_match_full",
    "result_match_reason",
    "latency_total",
    "tokens_per_second",
    "n_llm_calls",
)

DEFAULT_MAX_TOKENS = 256
GENERATION_TEMPERATURE = 0.0

_BASE_INSTRUCTION = """You are a DuckDB SQL generator.

Task:
Generate one valid DuckDB SQL query that answers the user question.

Rules:
- Return SQL only. No markdown, no code fences, no explanation.
- Use only the exact table name and columns present in the schema.

Schema:
{schema_context}
"""

# Static exemplars for few_shot_static. They cover the recurring energy
# patterns (whole-day filter, date RANGE with both bounds, kWh conversion,
# top-N) with parameter values chosen to differ from the gold sets.
ENERGY_STATIC_EXEMPLARS = """Examples:

Q: Berapa rata-rata tegangan pada tanggal 2 November 2010?
SQL:
SELECT AVG(Voltage) AS avg_voltage_v
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2010-11-02';

Q: Berapa rata-rata daya aktif antara 3 Mei 2010 dan 10 Mei 2010?
SQL:
SELECT AVG(Global_active_power) AS avg_global_active_power_kw
FROM electric_power
WHERE CAST(datetime AS DATE) BETWEEN DATE '2010-05-03' AND DATE '2010-05-10';

Q: Berapa total energi kWh sepanjang bulan Mei 2010?
SQL:
SELECT SUM(Global_active_power) / 60.0 AS total_energy_kwh
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2010
  AND EXTRACT(MONTH FROM datetime) = 5;

Q: Tampilkan 3 tanggal dengan rata-rata tegangan tertinggi pada tahun 2010.
SQL:
SELECT CAST(datetime AS DATE) AS day, AVG(Voltage) AS avg_voltage_v
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2010
GROUP BY day
ORDER BY avg_voltage_v DESC, day ASC
LIMIT 3;
"""

FINANCE_STATIC_EXEMPLARS = """Examples:

Q: Berapa harga penutupan rata-rata NVDA antara 1 Februari 2019 dan 28 Februari 2019?
SQL:
SELECT AVG(close) AS avg_close_usd
FROM stock_prices
WHERE ticker = 'NVDA'
  AND date BETWEEN DATE '2019-02-01' AND DATE '2019-02-28';

Q: Berapa volume perdagangan total TSLA sepanjang bulan Maret 2019?
SQL:
SELECT SUM(volume) AS total_volume
FROM stock_prices
WHERE ticker = 'TSLA'
  AND date BETWEEN DATE '2019-03-01' AND DATE '2019-03-31';

Q: Tampilkan 3 tanggal dengan harga penutupan GOOGL tertinggi selama periode dataset.
SQL:
SELECT date, close
FROM stock_prices
WHERE ticker = 'GOOGL'
ORDER BY close DESC, date ASC
LIMIT 3;
"""

STATIC_EXEMPLARS_BY_DOMAIN = {
    "energy": ENERGY_STATIC_EXEMPLARS,
    "finance": FINANCE_STATIC_EXEMPLARS,
}

_SCHEMA_LINKING_PROMPT = """You are a database expert doing schema linking.

Given the schema and the user question, list ONLY:
1. the table to query,
2. the columns needed (for SELECT, filters, grouping),
3. the filter conditions implied by the question (dates, thresholds,
   categories), written explicitly.

Answer as short plain-text bullet points. Do NOT write SQL yet.

Schema:
{schema_context}

Question: {question}

Schema links:"""

_DRAFT_PROMPT = """You are a DuckDB SQL generator.

Using the schema and the schema links below, generate one valid DuckDB SQL
query that answers the user question. Return SQL only, no markdown, no
explanation.

Schema:
{schema_context}

Schema links:
{schema_links}

Question: {question}

SQL:"""

_REFINE_PROMPT = """You are a DuckDB SQL reviewer.

Check the draft SQL against the question and the schema. Fix any mistakes:
- wrong or missing columns/table,
- wrong or INCOMPLETE date filters (a range needs BOTH the start AND end date),
- wrong aggregate function,
- missing unit conversion,
- invalid DuckDB syntax.

If the draft is already correct, return it unchanged. Return SQL only, no
markdown, no explanation.

Schema:
{schema_context}

Question: {question}

Draft SQL:
{draft_sql}

Final SQL:"""


@dataclass
class GenerationResult:
    """One strategy's SQL for one question, with generation telemetry."""

    sql: str
    n_llm_calls: int
    latency_seconds: float
    tokens_per_second: float | None
    error: str = ""


def _looks_like_sql(text: str) -> bool:
    return bool(text) and re.match(r"(?is)^\s*(SELECT|WITH)\b", text) is not None


def build_schema_context(domain: str) -> str:
    """The same compact DatasetProfile context the real pipeline uses."""
    return profile_to_compact_sql_context(load_dataset_profile(domain))


def build_zero_shot_prompt(question: str, schema_context: str) -> str:
    return (
        _BASE_INSTRUCTION.format(schema_context=schema_context)
        + f"\nQuestion: {question}\n\nSQL:"
    )


def build_few_shot_prompt(
    question: str, schema_context: str, domain: str = "energy"
) -> str:
    exemplars = STATIC_EXEMPLARS_BY_DOMAIN.get(
        domain, ENERGY_STATIC_EXEMPLARS
    )
    return (
        _BASE_INSTRUCTION.format(schema_context=schema_context)
        + "\n"
        + exemplars
        + f"\nQ: {question}\nSQL:"
    )


def generate_with_strategy(
    strategy: str,
    question: str,
    schema_context: str,
    ollama_tool: Any,
    domain: str = "energy",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> GenerationResult:
    """Generate SQL for one question with one strategy (raw extraction)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'")

    started = perf_counter()
    eval_count_total = 0.0
    eval_duration_total = 0.0
    n_calls = 0

    def call(prompt: str) -> str:
        nonlocal eval_count_total, eval_duration_total, n_calls
        response = ollama_tool.generate(
            prompt,
            temperature=GENERATION_TEMPERATURE,
            max_tokens=max_tokens,
        )
        n_calls += 1
        metrics = ollama_tool.get_last_metrics() or {}
        count = metrics.get("eval_count")
        duration = metrics.get("eval_duration")
        if isinstance(count, (int, float)) and isinstance(duration, (int, float)):
            eval_count_total += float(count)
            eval_duration_total += float(duration)
        return response

    try:
        if strategy == STRATEGY_ZERO_SHOT:
            response = call(build_zero_shot_prompt(question, schema_context))
            sql = extract_raw_model_sql(response)
        elif strategy == STRATEGY_FEW_SHOT:
            response = call(
                build_few_shot_prompt(question, schema_context, domain)
            )
            sql = extract_raw_model_sql(response)
        else:  # decomposed (DIN-SQL style)
            links = call(
                _SCHEMA_LINKING_PROMPT.format(
                    schema_context=schema_context, question=question
                )
            )
            draft = extract_raw_model_sql(
                call(
                    _DRAFT_PROMPT.format(
                        schema_context=schema_context,
                        schema_links=links.strip(),
                        question=question,
                    )
                )
            )
            refined = extract_raw_model_sql(
                call(
                    _REFINE_PROMPT.format(
                        schema_context=schema_context,
                        question=question,
                        draft_sql=draft,
                    )
                )
            )
            # The refine step may answer in prose; only accept it when it
            # actually looks like SQL, otherwise keep the draft.
            sql = refined if _looks_like_sql(refined) else draft
    except Exception as exc:
        return GenerationResult(
            sql="",
            n_llm_calls=n_calls,
            latency_seconds=perf_counter() - started,
            tokens_per_second=None,
            error=str(exc),
        )

    tokens_per_second = (
        eval_count_total / eval_duration_total
        if eval_duration_total > 0
        else None
    )
    return GenerationResult(
        sql=sql,
        n_llm_calls=n_calls,
        latency_seconds=perf_counter() - started,
        tokens_per_second=tokens_per_second,
    )


Logger = Callable[[str], None]


def run_prompting_comparison(
    questions: list[dict[str, Any]],
    strategies: tuple[str, ...],
    ollama_tool: Any,
    duckdb_tool: Any,
    domain: str = "energy",
    schema_context: str | None = None,
    logger: Logger = print,
) -> list[dict[str, Any]]:
    """Run every strategy over every question; returns flat comparison rows."""
    schema_context = schema_context or build_schema_context(domain)

    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        logger(f"[{strategy}] {len(questions)} questions")
        for question in questions:
            rows.append(
                _run_single(
                    strategy,
                    question,
                    schema_context,
                    ollama_tool,
                    duckdb_tool,
                    domain,
                )
            )
    return rows


def _run_single(
    strategy: str,
    question: dict[str, Any],
    schema_context: str,
    ollama_tool: Any,
    duckdb_tool: Any,
    domain: str,
) -> dict[str, Any]:
    question_id = str(question.get("id", ""))
    question_text = str(question.get("question", ""))
    errors: list[str] = []

    try:
        gold_sql = load_gold_sql(question["gold_sql_file"])
    except Exception as exc:
        gold_sql = ""
        errors.append(f"gold_sql_load: {exc}")

    generation = generate_with_strategy(
        strategy, question_text, schema_context, ollama_tool, domain
    )
    if generation.error:
        errors.append(f"generation: {generation.error}")

    if generation.sql:
        agent_result = execute_sql(duckdb_tool, generation.sql)
    else:
        agent_result = SqlExecutionResult(
            success=False,
            result_text="",
            error_message="Strategy produced no SQL",
        )
    if gold_sql:
        gold_result = execute_sql(duckdb_tool, gold_sql)
    else:
        gold_result = SqlExecutionResult(
            success=False, result_text="", error_message="Gold SQL not loaded"
        )

    if agent_result.error_message:
        errors.append(f"agent_sql: {agent_result.error_message}")
    if gold_result.error_message:
        errors.append(f"gold_sql: {gold_result.error_message}")

    numeric = compare_numeric_results(
        agent_result.numeric_value, gold_result.numeric_value
    )
    full = compare_result_sets(
        agent_result.result_text, gold_result.result_text
    )

    return {
        "config": strategy,
        "question_id": question_id,
        "question": question_text,
        "agent_sql": generation.sql,
        "gold_sql": gold_sql,
        "execution_success": agent_result.success,
        "agent_success": agent_result.success,
        "gold_success": gold_result.success,
        "numeric_match": numeric["numeric_match"],
        "absolute_error": numeric["absolute_error"],
        "relative_error": numeric["relative_error"],
        "error_message": " | ".join(errors),
        "result_match_full": full["result_match_full"],
        "result_match_reason": full["result_match_reason"],
        "latency_total": generation.latency_seconds,
        "tokens_per_second": (
            "" if generation.tokens_per_second is None
            else generation.tokens_per_second
        ),
        "n_llm_calls": generation.n_llm_calls,
    }
