"""Reproduce experiment suites from their manifests, one phase at a time.

Phases (match the development plan):
  1  Multi-model benchmark  - re-runs every model/suite recorded in the
     model_benchmark*/ manifests AFTER verifying the current gold dataset
     hashes still match the manifests (apple-to-apple guarantee).
  2  Error taxonomy         - re-classifies all eval/benchmark CSVs.
  3  Gold verification      - re-executes every gold SQL + pandas cross-check.
  4  System comparators     - prompting comparison (per model already run)
     and RAG retrieval eval (skips gracefully while gold is unlabeled).
  5  Checksums              - verify references/CHECKSUMS.txt against the
     current gold/eval files (use --write-checksums to (re)generate it).

Examples:
    python scripts/reproduce_all.py --phase 1 --dry-run
    python scripts/reproduce_all.py --phase 2 --phase 3
    python scripts/reproduce_all.py --phase 5 --write-checksums
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.model_benchmark import sha256_file


PYTHON = sys.executable
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
EXPERIMENTS_DIR = PROJECT_ROOT / "reports" / "experiments"
CHECKSUMS_PATH = PROJECT_ROOT / "references" / "CHECKSUMS.txt"

CHECKSUM_GLOBS = (
    ("references/sql_gold", "*.json"),
    ("references/sql_gold", "*.sql"),
    ("references/rag_gold", "*.json"),
)

SUITE_DEFAULT_QUESTIONS = {
    "sql_gold_v2": "references/sql_gold/energy_gold_questions_v2.json",
    "ablation": "references/sql_gold/energy_gold_questions_v2.json",
    "finance": "references/sql_gold/finance_gold_questions.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run experiment suites from their manifests."
    )
    parser.add_argument(
        "--phase",
        type=int,
        action="append",
        choices=(1, 2, 3, 4, 5),
        required=True,
        help="Phase to reproduce (repeatable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands and hash checks without executing anything.",
    )
    parser.add_argument(
        "--write-checksums",
        action="store_true",
        help="(Phase 5) regenerate references/CHECKSUMS.txt instead of verifying.",
    )
    return parser.parse_args()


def _run(command: list[str], dry_run: bool) -> bool:
    printable = " ".join(str(part) for part in command)
    if dry_run:
        print(f"  [dry-run] {printable}")
        return True
    print(f"  [run] {printable}")
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        return False
    return True


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Phase 1: model benchmark from manifests
# ---------------------------------------------------------------------------


def phase_1(dry_run: bool) -> bool:
    ok = True
    manifest_paths = sorted(EXPERIMENTS_DIR.glob("model_benchmark*/*/manifest.json"))
    if not manifest_paths:
        print("  No model_benchmark manifests found; nothing to reproduce.")
        return True

    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_key = str(manifest.get("model_key", manifest_path.parent.name))
        output_dir = manifest_path.parent.parent
        suites = manifest.get("suites", {})
        for suite_name, suite_entry in suites.items():
            dataset = suite_entry.get("dataset", {})
            questions_path = str(dataset.get("questions_path", ""))
            recorded_hash = str(dataset.get("questions_sha256", ""))
            repeats = int(suite_entry.get("repeats", 3))

            # Hash gate: refuse to call a rerun "reproduction" when the gold
            # set changed since the manifest was written.
            if questions_path and recorded_hash:
                current_file = PROJECT_ROOT / questions_path
                if not current_file.is_file():
                    print(
                        f"  HASH MISMATCH {model_key}/{suite_name}: "
                        f"{questions_path} no longer exists"
                    )
                    ok = False
                    continue
                current_hash = sha256_file(current_file)
                if current_hash != recorded_hash:
                    print(
                        f"  HASH MISMATCH {model_key}/{suite_name}: "
                        f"{questions_path} changed since the manifest "
                        f"({current_hash[:12]}... != {recorded_hash[:12]}...)"
                    )
                    ok = False
                    continue
                print(
                    f"  hash OK {model_key}/{suite_name}: {questions_path} "
                    f"({recorded_hash[:12]}...)"
                )

            command = [
                PYTHON,
                str(SCRIPTS_DIR / "run_model_benchmark.py"),
                "--models", model_key,
                "--suite", suite_name,
                "--repeats", str(repeats),
                "--output-dir", str(output_dir),
            ]
            if (
                questions_path
                and questions_path != SUITE_DEFAULT_QUESTIONS.get(suite_name)
            ):
                command += ["--questions-path", questions_path]
            if not _run(command, dry_run):
                ok = False

    summary_dirs = sorted(
        {path.parent.parent for path in manifest_paths}
    )
    for directory in summary_dirs:
        if not _run(
            [
                PYTHON,
                str(SCRIPTS_DIR / "summarize_model_benchmark.py"),
                "--input-dir", str(directory),
            ],
            dry_run,
        ):
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Phases 2-4: fixed script invocations
# ---------------------------------------------------------------------------


def phase_2(dry_run: bool) -> bool:
    return _run(
        [PYTHON, str(SCRIPTS_DIR / "run_error_taxonomy.py")], dry_run
    )


def phase_3(dry_run: bool) -> bool:
    return _run([PYTHON, str(SCRIPTS_DIR / "verify_gold_v3.py")], dry_run)


def phase_4(dry_run: bool) -> bool:
    ok = True
    comparison_dir = EXPERIMENTS_DIR / "prompting_comparison"
    model_keys = (
        sorted(path.name for path in comparison_dir.iterdir() if path.is_dir())
        if comparison_dir.is_dir()
        else []
    )
    if not model_keys:
        print("  No prompting_comparison results found; skipping that part.")
    for model_key in model_keys:
        if not _run(
            [
                PYTHON,
                str(SCRIPTS_DIR / "run_prompting_comparison.py"),
                "--model", model_key,
            ],
            dry_run,
        ):
            ok = False
    if not _run([PYTHON, str(SCRIPTS_DIR / "run_rag_eval.py")], dry_run):
        ok = False
    return ok


# ---------------------------------------------------------------------------
# Phase 5: checksums
# ---------------------------------------------------------------------------


def _checksum_files() -> list[Path]:
    files: list[Path] = []
    for directory, pattern in CHECKSUM_GLOBS:
        files.extend(sorted((PROJECT_ROOT / directory).glob(pattern)))
    return files


def phase_5(dry_run: bool, write: bool) -> bool:
    files = _checksum_files()
    if write:
        if dry_run:
            print(f"  [dry-run] would write {len(files)} checksums -> "
                  f"{_relative(CHECKSUMS_PATH)}")
            return True
        lines = [
            f"{sha256_file(path)}  {_relative(path)}" for path in files
        ]
        CHECKSUMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHECKSUMS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  Wrote {len(lines)} checksums -> {_relative(CHECKSUMS_PATH)}")
        return True

    if not CHECKSUMS_PATH.is_file():
        print(
            f"  {_relative(CHECKSUMS_PATH)} not found. Generate it with "
            "--write-checksums."
        )
        return False
    if dry_run:
        print(f"  [dry-run] would verify {_relative(CHECKSUMS_PATH)}")
        return True

    ok = True
    checked = 0
    for line in CHECKSUMS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        recorded_hash, _, rel_path = line.partition("  ")
        target = PROJECT_ROOT / rel_path
        if not target.is_file():
            print(f"  MISSING: {rel_path}")
            ok = False
            continue
        actual = sha256_file(target)
        checked += 1
        if actual != recorded_hash:
            print(f"  MISMATCH: {rel_path}")
            ok = False
    print(f"  Verified {checked} files against {_relative(CHECKSUMS_PATH)}: "
          + ("all OK" if ok else "PROBLEMS FOUND"))
    return ok


def main() -> int:
    args = parse_args()
    phases = sorted(set(args.phase))

    runners = {1: phase_1, 2: phase_2, 3: phase_3, 4: phase_4}
    all_ok = True
    for phase in phases:
        print(f"\n=== Phase {phase} ===")
        if phase == 5:
            ok = phase_5(args.dry_run, args.write_checksums)
        else:
            ok = runners[phase](args.dry_run)
        all_ok = all_ok and ok

    print("\nResult: " + ("OK" if all_ok else "PROBLEMS FOUND"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
