from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.finetune.dataset_builder import (
    Example,
    allocate_counts,
    build_dataset,
    read_jsonl,
    write_jsonl,
    write_outputs,
)
from local_agentic_analytics.finetune.teacher_client import FakeTeacherClient
from local_agentic_analytics.finetune.value_sampler import ENERGY_CHARTS


def test_allocate_counts_spreads_evenly():
    assert allocate_counts(12, ENERGY_CHARTS) == {chart: 2 for chart in ENERGY_CHARTS}
    counts = allocate_counts(14, ENERGY_CHARTS)
    assert sum(counts.values()) == 14
    assert max(counts.values()) - min(counts.values()) <= 1


def test_jsonl_round_trip(tmp_path):
    examples = [
        Example("instr", "chart_id: x\nvalue: 1", "narasi 1.0 kW", "x"),
        Example("instr", "chart_id: y\nvalue: 2", "narasi 2.0 kW", "y"),
    ]
    path = write_jsonl(tmp_path / "data.jsonl", examples)
    records = read_jsonl(path)
    assert records == [
        {"instruction": "instr", "input": "chart_id: x\nvalue: 1", "output": "narasi 1.0 kW"},
        {"instruction": "instr", "input": "chart_id: y\nvalue: 2", "output": "narasi 2.0 kW"},
    ]
    assert set(records[0]) == {"instruction", "input", "output"}


def test_build_dataset_with_fake_teacher_accepts_all_and_is_deterministic():
    result_a = build_dataset(
        FakeTeacherClient(seed=42), n=12, seed=42, concept_bank_text=""
    )
    result_b = build_dataset(
        FakeTeacherClient(seed=42), n=12, seed=42, concept_bank_text=""
    )

    assert result_a.total_generated == 12
    assert len(result_a.accepted) == 12
    assert result_a.rejected == []
    assert result_a.acceptance_rate == 1.0

    records_a = [example.to_record() for example in result_a.accepted]
    records_b = [example.to_record() for example in result_b.accepted]
    assert records_a == records_b


def test_build_dataset_split_sizes():
    result = build_dataset(
        FakeTeacherClient(seed=1), n=20, seed=1, concept_bank_text="", val_fraction=0.2
    )
    assert len(result.train) + len(result.val) == len(result.accepted)
    assert len(result.val) == round(len(result.accepted) * 0.2)
    assert len(result.train) >= len(result.val)


def test_write_outputs_creates_all_files(tmp_path):
    result = build_dataset(
        FakeTeacherClient(seed=3), n=6, seed=3, concept_bank_text=""
    )
    paths = write_outputs(tmp_path, result)
    assert paths["train"].exists()
    assert paths["val"].exists()
    assert paths["rejected"].exists()

    rejected_text = paths["rejected"].read_text(encoding="utf-8")
    assert rejected_text.splitlines()[0] == "chart_id,index,reasons,narrative"
