# local-agentic-analytics

Local agentic data analytics system for a final project. The project is designed to run on a modest laptop using a local small language model, DuckDB for structured analytics, and later ChromaDB for retrieval-augmented workflows.

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
|-- scripts/
|-- src/
|   `-- local_agentic_analytics/
|       |-- agents/
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

## Development Principles

- Keep workflows sequential and lightweight.
- Use one local model through Ollama for multiple agent roles.
- Start with DuckDB text-to-SQL before adding RAG.
- Avoid loading large datasets with pandas.
- Keep modules small and add tests for important tools.

## Status

This repository currently contains the initial project skeleton only. Agent implementations, graph workflows, and analytics tools will be added incrementally.
