"""CI-safe checks for the v2 gold SQL benchmark.

Loads references/sql_gold/energy_gold_questions_v2.json and asserts that every
gold .sql file parses and executes. Execution runs against a SYNTHETIC
in-memory DuckDB table (no dependency on the real analytics.duckdb), so this
test is safe to run in CI. Correctness of the computed values against the real
database is verified separately by scripts/verify_gold_v2.py.
"""

from pathlib import Path
import json
import sys

import duckdb
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.sql_gold_eval import load_gold_questions

MANIFEST_PATH = (
    PROJECT_ROOT / "references" / "sql_gold" / "energy_gold_questions_v2.json"
)
REQUIRED_TAGS = ("category", "difficulty", "resolver_eligible", "result_shape")
VALID_SHAPES = {"scalar", "row", "table"}
VALID_RESOLVER = {"yes", "no"}


@pytest.fixture(scope="module")
def synthetic_connection() -> duckdb.DuckDBPyConnection:
    """Build an in-memory electric_power table covering the real date span.

    Minute-level rows are sampled every 11 minutes from 2006-12-16 to
    2010-11-26 so that day, hour, month, week and year filters in the gold
    SQL all match some rows. Values are deterministic pseudo-random within the
    real column ranges; they are NOT meant to match the real data, only to let
    every gold query execute.
    """
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE electric_power AS
        SELECT
            ts AS datetime,
            0.076 + (hash(epoch(ts)) % 1105) / 100.0 AS Global_active_power,
            (hash(epoch(ts)) % 1391) / 1000.0 AS Global_reactive_power,
            223.2 + (hash(epoch(ts)) % 3096) / 100.0 AS Voltage,
            0.2 + (hash(epoch(ts)) % 483) / 10.0 AS Global_intensity,
            CAST(hash(epoch(ts)) % 89 AS DOUBLE) AS Sub_metering_1,
            CAST(hash(epoch(ts)) % 81 AS DOUBLE) AS Sub_metering_2,
            CAST(hash(epoch(ts)) % 32 AS DOUBLE) AS Sub_metering_3
        FROM (
            SELECT UNNEST(generate_series(
                TIMESTAMP '2006-12-16 17:24:00',
                TIMESTAMP '2010-11-26 21:02:00',
                INTERVAL 11 MINUTE
            )) AS ts
        );
        """
    )
    yield con
    con.close()


def _load_manifest() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_loads_with_required_base_fields():
    # load_gold_questions enforces id/question/gold_sql_file/expected_unit.
    questions = load_gold_questions(MANIFEST_PATH)
    assert len(questions) >= 30
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "question ids must be unique"


def test_manifest_has_valid_v2_tags_and_existing_sql_files():
    for q in _load_manifest():
        for tag in REQUIRED_TAGS:
            assert q.get(tag), f"{q['id']} missing tag '{tag}'"
        assert q["result_shape"] in VALID_SHAPES, q["id"]
        assert q["resolver_eligible"] in VALID_RESOLVER, q["id"]
        sql_path = PROJECT_ROOT / q["gold_sql_file"]
        assert sql_path.is_file(), f"missing gold SQL file for {q['id']}"
        assert sql_path.read_text(encoding="utf-8").strip(), f"empty SQL {q['id']}"


@pytest.mark.parametrize("question", _load_manifest(), ids=lambda q: q["id"])
def test_every_gold_sql_parses_and_executes(question, synthetic_connection):
    sql = (PROJECT_ROOT / question["gold_sql_file"]).read_text(encoding="utf-8")

    result = synthetic_connection.execute(sql).fetchdf()

    assert isinstance(result, pd.DataFrame)
    # A declared scalar must be exactly one cell when it returns anything.
    if question["result_shape"] == "scalar" and not result.empty:
        assert result.shape == (1, 1), (
            f"{question['id']} declared scalar but returned {result.shape}"
        )
