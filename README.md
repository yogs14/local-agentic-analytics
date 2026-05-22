# local-agentic-analytics

Local agentic data analytics system for a final project. The system runs on a modest local laptop with 8GB RAM and GTX 1650 4GB, so the implementation is intentionally lightweight, sequential, and modular.

The current focus is structured analytics with DuckDB and a local Ollama small language model. ChromaDB/RAG support exists as an early-stage document retrieval module and is kept separate from the main DuckDB workflow.

## Current Status

Implemented:

- DuckDB ingestion for the Individual Household Electric Power Consumption dataset.
- Lightweight DuckDB tool for schema lookup, CSV registration, SQL execution, and samples.
- Lightweight Ollama client using local API.
- Text-to-SQL agent for DuckDB SQL generation.
- SQL repair agent with one repair attempt per query.
- Reporter agent for short Indonesian answers.
- Sequential text-to-SQL workflow, without parallel agents.
- ChromaDB tool for persistent local document retrieval.
- Small dummy RAG build and retrieval scripts.
- CSV experiment logging for workflow runs.
- Unit tests for core tools, agents, workflow, and logging.

Not implemented yet:

- LangGraph orchestration.
- Full production RAG ingestion.
- Multi-table schema selection.
- Advanced evaluation metrics.
- UI or API server.

## Project Structure

```text
local-agentic-analytics/
|-- configs/
|-- data/
|   |-- raw/
|   `-- processed/
|-- databases/
|   |-- chromadb/
|   `-- duckdb/
|-- notebooks/
|-- reports/
|   `-- experiments/
|-- scripts/
|-- src/
|   `-- local_agentic_analytics/
|       |-- agents/
|       |-- core/
|       |-- data/
|       |-- evaluation/
|       |-- graph/
|       |-- prompts/
|       `-- tools/
`-- tests/
```

## Setup on Windows

Use Python 3.10 or 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If Python 3.11 is not installed, use Python 3.10:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If the `py` launcher is unavailable, use:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

## Ollama Setup

Install and start Ollama, then make sure the configured model is available.

```powershell
ollama pull gemma2:2b
ollama list
```

The default config uses:

```yaml
model:
  provider: ollama
  name: ${OLLAMA_MODEL}
  context_window: 2048
  num_gpu: 0
```

`num_gpu: 0` is intentional for stability on GTX 1650 4GB. It avoids CUDA out-of-memory crashes by running generation on CPU. It is slower, but more reliable on the target laptop.

## Dataset Ingestion

Place the UCI Individual Household Electric Power Consumption file here:

```text
data/raw/energy/household_power_consumption.txt
```

Then run:

```powershell
python scripts/ingest_energy.py
```

This creates or replaces the DuckDB table:

```text
databases/duckdb/analytics.duckdb
table: electric_power
```

The ingestion uses DuckDB directly and avoids pandas full-dataset processing.

## Text-to-SQL Workflow

Run a local analytics query:

```powershell
python scripts/run_workflow.py "Berapa rata-rata konsumsi daya aktif pada tanggal 16 Desember 2006?"
```

The script prints:

- generated SQL
- repaired SQL, if any
- compact query result
- final Indonesian answer
- latency per step
- success or failure status

Workflow steps:

1. Read `electric_power` schema from DuckDB.
2. Generate DuckDB SQL with the local Ollama model.
3. Execute SQL with DuckDB.
4. If SQL fails, repair once.
5. Execute repaired SQL once.
6. Generate a short Indonesian report.
7. Log latency and status.

## RAG / ChromaDB

Build a small dummy ChromaDB collection:

```powershell
python scripts/build_chromadb.py
```

Smoke test retrieval only:

```powershell
python scripts/test_rag_retrieval.py
```

Run a simple RAG query with ChromaDB retrieval and Ollama answer generation:

```powershell
python scripts/run_rag_query.py "Apa satuan Global_active_power?"
```

RAG is currently experimental and uses small dummy documents. It does not use DuckDB.

## Evaluation Logs

Workflow runs append logs to:

```text
reports/experiments/runs.csv
```

Logged columns include:

- timestamp
- user_query
- generated_sql
- repaired_sql
- success
- error_message
- latency_total
- latency_sql_generation
- latency_sql_execution
- latency_reporting

## Run Tests

```powershell
python -m pytest tests
```

Some ChromaDB/Pydantic deprecation warnings may appear. They are dependency warnings, not test failures.

## Development Principles

- Keep workflows sequential and lightweight.
- Use one local model through Ollama for multiple agent roles.
- Do not create parallel multi-agent execution.
- Do not use different models for each agent.
- Use DuckDB for structured data.
- Use ChromaDB only for RAG/document retrieval.
- Avoid loading large datasets with pandas.
- Keep files small and modular.
- Add tests or validation scripts for important tools and features.
- Avoid heavy dependencies such as TensorFlow, custom GPU Torch builds, AutoGen, or CrewAI unless there is a clear reason.

## Notes

This project is optimized for gradual development. The current stable path is DuckDB text-to-SQL. RAG is available as a separate early-stage module and should be expanded later after the structured analytics workflow is reliable.
