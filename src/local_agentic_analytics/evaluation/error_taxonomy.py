"""Error taxonomy for failed text-to-SQL benchmark rows.

Classifies why an agent answer failed against its gold SQL using exclusive,
ordered categories — the FIRST matching category wins (one primary error per
question):

1. ``syntax_error``       - SQL missing or not valid DuckDB.
2. ``schema_linking``     - wrong/hallucinated table or column (binder errors,
                            identifiers outside the schema vocabulary).
3. ``unit_conversion``    - arithmetic conversion constants differ (e.g. the
                            missing ``/60.0`` for kWh).
4. ``date_filter``        - date/year literals in filters differ.
5. ``aggregation_choice`` - different aggregate functions (AVG vs SUM vs MAX).
6. ``grouping_logic``     - GROUP BY / HAVING / DATE_TRUNC / subquery /
                            window structure differs.
7. ``nl_understanding_id``- SQL is valid and structurally plausible but uses a
                            different (existing) measure column, i.e. it
                            answers a different question. Always flagged
                            ``needs_manual_review``.
8. ``output_shape``       - the gold values are present in the agent result
                            but the shape/format differs, so the numeric
                            compare failed.
9. ``other``              - everything else. Always ``needs_manual_review``.

Heuristics parse both SQLs into ASTs with ``sqlglot`` (DuckDB dialect) and
compare structural features. Ambiguous outcomes are flagged
``needs_manual_review`` for the human pass; this module never guesses silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import sqlglot
from sqlglot import expressions as exp

from local_agentic_analytics.evaluation.result_comparison import (
    compare_result_sets,
    parse_result_records,
)


CATEGORY_SYNTAX = "syntax_error"
CATEGORY_SCHEMA = "schema_linking"
CATEGORY_UNIT = "unit_conversion"
CATEGORY_DATE = "date_filter"
CATEGORY_AGGREGATION = "aggregation_choice"
CATEGORY_GROUPING = "grouping_logic"
CATEGORY_NL_UNDERSTANDING = "nl_understanding_id"
CATEGORY_OUTPUT_SHAPE = "output_shape"
CATEGORY_OTHER = "other"

TAXONOMY_CATEGORIES = (
    CATEGORY_SYNTAX,
    CATEGORY_SCHEMA,
    CATEGORY_UNIT,
    CATEGORY_DATE,
    CATEGORY_AGGREGATION,
    CATEGORY_GROUPING,
    CATEGORY_NL_UNDERSTANDING,
    CATEGORY_OUTPUT_SHAPE,
    CATEGORY_OTHER,
)

_SYNTAX_ERROR_SIGNATURES = (
    "parser error",
    "syntax error",
    "unexpected token",
)
_SCHEMA_ERROR_SIGNATURES = (
    "binder error",
    "catalog error",
    "does not exist",
    "not found in from clause",
    "referenced column",
)

_DATE_LITERAL_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMERIC_TOLERANCE = 1e-6

_KNOWN_ANONYMOUS_AGGREGATES = {
    "avg",
    "sum",
    "min",
    "max",
    "count",
    "median",
    "stddev",
    "stddev_samp",
    "var_samp",
    "variance",
}


@dataclass(frozen=True)
class TaxonomyResult:
    """Primary error category for one failed row."""

    category: str
    needs_manual_review: bool
    evidence: str


@dataclass
class SqlFeatures:
    """Structural features extracted from one SQL statement's AST."""

    parse_ok: bool = False
    error: str = ""
    tables: set[str] = field(default_factory=set)
    columns: set[str] = field(default_factory=set)
    aggregates: set[str] = field(default_factory=set)
    date_literals: set[str] = field(default_factory=set)
    year_literals: set[str] = field(default_factory=set)
    arithmetic_constants: set[str] = field(default_factory=set)
    has_group_by: bool = False
    group_expressions: set[str] = field(default_factory=set)
    has_having: bool = False
    has_subquery: bool = False
    has_window: bool = False
    select_width: int = 0


def extract_sql_features(sql: str) -> SqlFeatures:
    """Parse one SQL statement (DuckDB dialect) into comparable features."""
    features = SqlFeatures()
    if not sql or not sql.strip():
        features.error = "empty SQL"
        return features

    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except Exception as exc:
        features.error = f"sqlglot parse failed: {exc}"
        return features
    if tree is None:
        features.error = "sqlglot returned no AST"
        return features

    features.parse_ok = True
    features.tables = {
        table.name.lower() for table in tree.find_all(exp.Table) if table.name
    }
    features.columns = {
        column.name.lower() for column in tree.find_all(exp.Column) if column.name
    }
    features.aggregates = _extract_aggregates(tree)
    features.date_literals = set(_DATE_LITERAL_PATTERN.findall(sql))
    features.year_literals = _extract_year_literals(tree)
    features.arithmetic_constants = _extract_arithmetic_constants(tree)

    group = tree.find(exp.Group)
    features.has_group_by = group is not None
    if group is not None:
        features.group_expressions = {
            expression.sql(dialect="duckdb").lower()
            for expression in group.expressions
        }
    features.has_having = tree.find(exp.Having) is not None
    selects = list(tree.find_all(exp.Select))
    features.has_subquery = len(selects) > 1 or tree.find(exp.Subquery) is not None
    features.has_window = tree.find(exp.Window) is not None
    if isinstance(tree, exp.Select):
        features.select_width = len(tree.expressions)
    elif selects:
        features.select_width = len(selects[0].expressions)
    return features


def _extract_aggregates(tree: exp.Expression) -> set[str]:
    aggregates = {
        type(node).__name__.lower() for node in tree.find_all(exp.AggFunc)
    }
    for node in tree.find_all(exp.Anonymous):
        name = str(node.name or "").lower()
        if name in _KNOWN_ANONYMOUS_AGGREGATES:
            aggregates.add(name)
    return aggregates


def _extract_year_literals(tree: exp.Expression) -> set[str]:
    years: set[str] = set()
    for literal in tree.find_all(exp.Literal):
        if literal.is_string:
            continue
        text = str(literal.this)
        if re.fullmatch(r"(19|20)\d{2}", text):
            years.add(text)
    return years


def _extract_arithmetic_constants(tree: exp.Expression) -> set[str]:
    """Constants used in multiplication/division (unit-conversion factors)."""
    constants: set[str] = set()
    for node in tree.find_all(exp.Mul, exp.Div):
        for operand in (node.this, node.expression):
            if isinstance(operand, exp.Literal) and not operand.is_string:
                constants.add(_normalize_constant(str(operand.this)))
    return constants


def _normalize_constant(text: str) -> str:
    try:
        value = float(text)
    except ValueError:
        return text
    if value.is_integer():
        return str(int(value))
    return repr(value)


def build_schema_vocabulary(gold_sqls: list[str]) -> dict[str, set[str]]:
    """Collect the table/column vocabulary from every gold SQL in a set.

    Identifiers used by an agent that fall outside this vocabulary are treated
    as schema-linking failures (hallucinated columns/tables).
    """
    tables: set[str] = set()
    columns: set[str] = set()
    for sql in gold_sqls:
        features = extract_sql_features(sql)
        tables.update(features.tables)
        columns.update(features.columns)
    return {"tables": tables, "columns": columns}


def is_failed_row(row: Mapping[str, Any]) -> bool:
    """Decide whether a benchmark CSV row counts as a failure to classify.

    A row fails when the agent SQL did not execute, or it executed but neither
    the legacy numeric compare nor the row-set compare found a match. Rows
    where the GOLD SQL itself failed are not agent failures (handled by the
    caller as ``gold_error``).
    """
    if not _cell_bool(row.get("gold_success")):
        return False

    executed = _cell_bool(
        row.get("execution_success", row.get("agent_success"))
    )
    if not executed:
        return True

    numeric_match = row.get("numeric_match", "")
    result_full = row.get("result_match_full", "")
    if _cell_bool(numeric_match) or _cell_bool(result_full):
        return False
    if numeric_match == "" and result_full == "":
        # Executed but nothing was comparable (e.g. legacy-only rows with a
        # non-scalar result): treat as failed so it gets classified/reviewed.
        return True
    return True


def _cell_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def classify_failure(
    agent_sql: str,
    gold_sql: str,
    error_message: str = "",
    expected_unit: str = "",
    agent_result_text: str | None = None,
    gold_result_text: str | None = None,
    vocabulary: dict[str, set[str]] | None = None,
) -> TaxonomyResult:
    """Assign the single primary error category (first match wins)."""
    lowered_error = (error_message or "").lower()

    # 1. syntax_error ------------------------------------------------------
    if not agent_sql or not agent_sql.strip():
        return TaxonomyResult(
            CATEGORY_SYNTAX, False, "agent produced no SQL"
        )
    if any(signature in lowered_error for signature in _SYNTAX_ERROR_SIGNATURES):
        return TaxonomyResult(
            CATEGORY_SYNTAX, False, "DuckDB reported a parser/syntax error"
        )

    agent = extract_sql_features(agent_sql)
    gold = extract_sql_features(gold_sql)

    if not agent.parse_ok:
        return TaxonomyResult(
            CATEGORY_SYNTAX,
            False,
            f"agent SQL does not parse ({agent.error})",
        )

    degraded = not gold.parse_ok  # gold should always parse; be safe.

    # 2. schema_linking ----------------------------------------------------
    if any(signature in lowered_error for signature in _SCHEMA_ERROR_SIGNATURES):
        return TaxonomyResult(
            CATEGORY_SCHEMA, False, "DuckDB reported a binder/catalog error"
        )
    if vocabulary is not None:
        unknown_tables = agent.tables - vocabulary.get("tables", set())
        unknown_columns = agent.columns - vocabulary.get("columns", set())
        if unknown_tables or unknown_columns:
            return TaxonomyResult(
                CATEGORY_SCHEMA,
                False,
                "identifiers outside schema vocabulary: "
                f"tables={sorted(unknown_tables)}, "
                f"columns={sorted(unknown_columns)}",
            )

    # 3. unit_conversion ---------------------------------------------------
    if not degraded and agent.arithmetic_constants != gold.arithmetic_constants:
        return TaxonomyResult(
            CATEGORY_UNIT,
            False,
            "conversion constants differ: "
            f"agent={sorted(agent.arithmetic_constants)}, "
            f"gold={sorted(gold.arithmetic_constants)}"
            + (f" (expected_unit={expected_unit})" if expected_unit else ""),
        )

    # 4. date_filter -------------------------------------------------------
    if not degraded and (
        agent.date_literals != gold.date_literals
        or agent.year_literals != gold.year_literals
    ):
        return TaxonomyResult(
            CATEGORY_DATE,
            False,
            "date/year literals differ: "
            f"agent={sorted(agent.date_literals | agent.year_literals)}, "
            f"gold={sorted(gold.date_literals | gold.year_literals)}",
        )

    # 5. aggregation_choice ------------------------------------------------
    if not degraded and agent.aggregates != gold.aggregates:
        return TaxonomyResult(
            CATEGORY_AGGREGATION,
            False,
            f"aggregates differ: agent={sorted(agent.aggregates)}, "
            f"gold={sorted(gold.aggregates)}",
        )

    # 6. grouping_logic ----------------------------------------------------
    if not degraded and (
        agent.has_group_by != gold.has_group_by
        or agent.group_expressions != gold.group_expressions
        or agent.has_having != gold.has_having
        or agent.has_subquery != gold.has_subquery
        or agent.has_window != gold.has_window
    ):
        return TaxonomyResult(
            CATEGORY_GROUPING,
            False,
            "grouping structure differs: "
            f"group_by agent={sorted(agent.group_expressions)} vs "
            f"gold={sorted(gold.group_expressions)}, "
            f"having={agent.has_having}/{gold.has_having}, "
            f"subquery={agent.has_subquery}/{gold.has_subquery}, "
            f"window={agent.has_window}/{gold.has_window}",
        )

    # 7. nl_understanding_id (always manual review) -------------------------
    if not degraded and agent.columns != gold.columns:
        return TaxonomyResult(
            CATEGORY_NL_UNDERSTANDING,
            True,
            "structure matches but different (existing) columns were used: "
            f"agent={sorted(agent.columns)}, gold={sorted(gold.columns)} "
            "- likely misread the Indonesian question",
        )

    # 8. output_shape --------------------------------------------------------
    overlap = _gold_values_covered_by_agent(agent_result_text, gold_result_text)
    if overlap is True:
        return TaxonomyResult(
            CATEGORY_OUTPUT_SHAPE,
            False,
            "gold values are all present in the agent result; only the "
            "shape/format differs",
        )

    # 9. other (always manual review) ---------------------------------------
    evidence = "no heuristic matched"
    if degraded:
        evidence = f"gold SQL failed to parse ({gold.error}); " + evidence
    return TaxonomyResult(CATEGORY_OTHER, True, evidence)


def _gold_values_covered_by_agent(
    agent_result_text: str | None,
    gold_result_text: str | None,
) -> bool | None:
    """True when every numeric gold value appears in the agent result."""
    agent_rows = parse_result_records(agent_result_text)
    gold_rows = parse_result_records(gold_result_text)
    if agent_rows is None or gold_rows is None:
        return None

    gold_numbers = _flatten_numbers(gold_rows)
    agent_numbers = _flatten_numbers(agent_rows)
    if not gold_numbers:
        return None

    remaining = list(agent_numbers)
    for gold_number in gold_numbers:
        index = _find_close(remaining, gold_number)
        if index is None:
            return False
        remaining.pop(index)
    return True


def _flatten_numbers(rows: list[list[Any]]) -> list[float]:
    numbers: list[float] = []
    for row in rows:
        for cell in row:
            if isinstance(cell, bool):
                continue
            if isinstance(cell, (int, float)):
                numbers.append(float(cell))
    return numbers


def _find_close(values: list[float], target: float) -> int | None:
    for index, value in enumerate(values):
        difference = abs(value - target)
        if difference <= _NUMERIC_TOLERANCE:
            return index
        if target != 0 and difference / abs(target) <= _NUMERIC_TOLERANCE:
            return index
    return None


# ---------------------------------------------------------------------------
# Benchmark-row driver (CSV rows from sql_gold_eval / ablation_eval / harness)
# ---------------------------------------------------------------------------

TAXONOMY_ROW_COLUMNS = (
    "source_file",
    "suite",
    "model",
    "run_index",
    "config",
    "question_id",
    "question",
    "category",
    "needs_manual_review",
    "evidence",
    "agent_sql",
    "gold_sql",
    "error_message",
)

GOLD_ERROR_LABEL = "gold_error"
LATE_MATCH_LABEL = "late_match_result_full"

# Signature: sql -> (success, result_text, error_message). Used to re-execute
# SQL for rows whose CSV format does not store result texts (ablation rows).
SqlExecutor = Any


def classify_benchmark_rows(
    rows: list[Mapping[str, Any]],
    questions_by_id: Mapping[str, Mapping[str, Any]],
    vocabulary: dict[str, set[str]] | None = None,
    sql_executor: SqlExecutor | None = None,
    gold_sql_loader: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify every failed row from one benchmark CSV.

    Returns ``(classified_rows, gold_error_rows, late_match_rows)``:

    - ``gold_error_rows``: the GOLD SQL itself failed — an agent-independent
      dataset problem, never classified as an agent error.
    - ``late_match_rows``: rows from legacy CSVs (without the
      ``result_match_full`` column) whose results actually MATCH under the
      row-set comparison — they were only "failures" because the legacy scalar
      metric could not compare them. They are successes, not errors.
    """
    metadata = dict(metadata or {})
    classified: list[dict[str, Any]] = []
    gold_errors: list[dict[str, Any]] = []
    late_matches: list[dict[str, Any]] = []

    for row in rows:
        question_id = str(row.get("question_id", ""))
        question_info = questions_by_id.get(question_id, {})

        if not _cell_bool(row.get("gold_success")):
            gold_errors.append(
                _output_row(
                    row,
                    metadata,
                    category=GOLD_ERROR_LABEL,
                    needs_manual_review=True,
                    evidence="gold SQL itself failed to execute",
                    gold_sql=str(row.get("gold_sql", "")),
                )
            )
            continue

        if not is_failed_row(row):
            continue

        gold_sql = str(row.get("gold_sql", "") or "")
        if not gold_sql and gold_sql_loader is not None:
            gold_sql_file = str(question_info.get("gold_sql_file", ""))
            if gold_sql_file:
                try:
                    gold_sql = gold_sql_loader(gold_sql_file)
                except Exception:
                    gold_sql = ""

        agent_sql = str(row.get("agent_sql", "") or "")
        agent_result_text = row.get("agent_result") or None
        gold_result_text = row.get("gold_result") or None
        error_message = str(row.get("error_message", "") or "")

        executed = _cell_bool(
            row.get("execution_success", row.get("agent_success"))
        )
        if (
            sql_executor is not None
            and executed
            and agent_sql
            and gold_sql
            and (agent_result_text is None or gold_result_text is None)
        ):
            agent_result_text = _safe_execute(sql_executor, agent_sql)
            gold_result_text = _safe_execute(sql_executor, gold_sql)

        # Legacy rows have no result_match_full cell; if their results turn
        # out to fully match under the row-set comparison, they are successes
        # the legacy scalar metric simply could not see - not errors.
        if executed and row.get("result_match_full", "") == "":
            late_comparison = compare_result_sets(
                agent_result_text, gold_result_text
            )
            if late_comparison["result_match_full"] is True:
                late_matches.append(
                    _output_row(
                        row,
                        metadata,
                        category=LATE_MATCH_LABEL,
                        needs_manual_review=False,
                        evidence=(
                            "results fully match under row-set comparison; "
                            "only the legacy scalar metric skipped this row"
                        ),
                        gold_sql=gold_sql,
                    )
                )
                continue

        result = classify_failure(
            agent_sql=agent_sql,
            gold_sql=gold_sql,
            error_message=error_message,
            expected_unit=str(question_info.get("expected_unit", "")),
            agent_result_text=agent_result_text,
            gold_result_text=gold_result_text,
            vocabulary=vocabulary,
        )
        classified.append(
            _output_row(
                row,
                metadata,
                category=result.category,
                needs_manual_review=result.needs_manual_review,
                evidence=result.evidence,
                gold_sql=gold_sql,
            )
        )

    return classified, gold_errors, late_matches


def _safe_execute(sql_executor: SqlExecutor, sql: str) -> str | None:
    try:
        success, result_text, _error = sql_executor(sql)
    except Exception:
        return None
    return result_text if success else None


def _output_row(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
    category: str,
    needs_manual_review: bool,
    evidence: str,
    gold_sql: str,
) -> dict[str, Any]:
    return {
        "source_file": str(metadata.get("source_file", "")),
        "suite": str(metadata.get("suite", "")),
        "model": str(metadata.get("model", "")),
        "run_index": str(metadata.get("run_index", "")),
        "config": str(row.get("config", "")),
        "question_id": str(row.get("question_id", "")),
        "question": str(row.get("question", "")),
        "category": category,
        "needs_manual_review": needs_manual_review,
        "evidence": evidence,
        "agent_sql": str(row.get("agent_sql", "")),
        "gold_sql": gold_sql,
        "error_message": str(row.get("error_message", "")),
    }


def distribution_by_group(
    classified_rows: list[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    """Count categories per group (ablation config, or model, or suite)."""
    distribution: dict[str, dict[str, int]] = {}
    for row in classified_rows:
        group = _group_key(row)
        category = str(row.get("category", CATEGORY_OTHER))
        distribution.setdefault(group, {})
        distribution[group][category] = (
            distribution[group].get(category, 0) + 1
        )
    return distribution


def _group_key(row: Mapping[str, Any]) -> str:
    config = str(row.get("config", "") or "")
    model = str(row.get("model", "") or "")
    if config and model:
        return f"{model}/{config}"
    if config:
        return config
    if model:
        return model
    return str(row.get("suite", "") or row.get("source_file", "") or "-")
