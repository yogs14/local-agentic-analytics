"""Orchestrate synthetic dataset generation, validation, and serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
import random
from typing import Any

from local_agentic_analytics.finetune.concept_bank import load_concept_bank_text
from local_agentic_analytics.finetune.teacher_client import TeacherClient
from local_agentic_analytics.finetune.teacher_prompt import (
    DEFAULT_INSTRUCTION,
    build_teacher_prompt,
)
from local_agentic_analytics.finetune.validators import validate_narrative
from local_agentic_analytics.finetune.value_sampler import (
    CHART_TITLES,
    ENERGY_CHARTS,
    UNIT_REQUIRED,
    format_stats_block,
    sample_variant,
)


DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 384


@dataclass
class Example:
    instruction: str
    input: str
    output: str
    chart_id: str

    def to_record(self) -> dict[str, str]:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }


@dataclass
class Rejection:
    chart_id: str
    index: int
    reasons: list[str]
    narrative: str


@dataclass
class BuildResult:
    accepted: list[Example]
    rejected: list[Rejection]
    train: list[Example]
    val: list[Example]
    total_generated: int
    per_chart_accepted: dict[str, int] = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        if self.total_generated == 0:
            return 0.0
        return len(self.accepted) / self.total_generated


def allocate_counts(total: int, charts: tuple[str, ...]) -> dict[str, int]:
    """Spread ``total`` variants across ``charts`` as evenly as possible."""
    if total <= 0 or not charts:
        return {chart: 0 for chart in charts}
    base, remainder = divmod(total, len(charts))
    counts: dict[str, int] = {}
    for position, chart in enumerate(charts):
        counts[chart] = base + (1 if position < remainder else 0)
    return counts


def build_dataset(
    client: TeacherClient,
    *,
    n: int,
    seed: int = 42,
    charts: tuple[str, ...] = ENERGY_CHARTS,
    concept_bank_text: str | None = None,
    instruction: str = DEFAULT_INSTRUCTION,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    val_fraction: float = 0.2,
    min_concept_terms: int = 2,
) -> BuildResult:
    """Generate, validate, and split a synthetic dataset.

    Deterministic for a fixed ``seed`` (assuming a deterministic ``client``).
    """
    if concept_bank_text is None:
        concept_bank_text = load_concept_bank_text()

    rng = random.Random(seed)
    counts = allocate_counts(n, charts)

    accepted: list[Example] = []
    rejected: list[Rejection] = []
    per_chart_accepted: dict[str, int] = {chart: 0 for chart in charts}
    total_generated = 0

    for chart_id in charts:
        for index in range(counts[chart_id]):
            total_generated += 1
            variant = sample_variant(chart_id, rng)
            block = format_stats_block(variant)
            prompt = build_teacher_prompt(
                block,
                chart_title=CHART_TITLES.get(chart_id, chart_id),
                concept_bank_text=concept_bank_text,
                instruction=instruction,
            )
            try:
                narrative = client.generate(
                    prompt, temperature=temperature, max_tokens=max_tokens
                )
            except Exception as exc:  # noqa: BLE001 - log and continue the batch
                rejected.append(
                    Rejection(
                        chart_id=chart_id,
                        index=index,
                        reasons=[f"teacher_error:{type(exc).__name__}:{exc}"],
                        narrative="",
                    )
                )
                continue
            result = validate_narrative(
                narrative,
                block,
                chart_id,
                require_unit=UNIT_REQUIRED.get(chart_id, True),
                min_concept_terms=min_concept_terms,
            )
            if result.ok:
                accepted.append(
                    Example(
                        instruction=instruction,
                        input=block,
                        output=narrative.strip(),
                        chart_id=chart_id,
                    )
                )
                per_chart_accepted[chart_id] += 1
            else:
                rejected.append(
                    Rejection(
                        chart_id=chart_id,
                        index=index,
                        reasons=result.reasons,
                        narrative=narrative.strip(),
                    )
                )

    train, val = _split(accepted, val_fraction, seed)
    return BuildResult(
        accepted=accepted,
        rejected=rejected,
        train=train,
        val=val,
        total_generated=total_generated,
        per_chart_accepted=per_chart_accepted,
    )


def _split(
    examples: list[Example], val_fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    val_size = int(round(len(shuffled) * val_fraction))
    # Keep at least one training example when any data exists.
    val_size = min(val_size, max(len(shuffled) - 1, 0))
    val = shuffled[:val_size]
    train = shuffled[val_size:]
    return train, val


def write_jsonl(path: str | Path, examples: list[Example]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_record(), ensure_ascii=False))
            handle.write("\n")
    return out


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_rejected_csv(path: str | Path, rejections: list[Rejection]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["chart_id", "index", "reasons", "narrative"])
        for rejection in rejections:
            writer.writerow(
                [
                    rejection.chart_id,
                    rejection.index,
                    "; ".join(rejection.reasons),
                    rejection.narrative.replace("\n", " "),
                ]
            )
    return out


def write_outputs(out_dir: str | Path, result: BuildResult) -> dict[str, Path]:
    """Write train.jsonl, val.jsonl, and rejected.csv into ``out_dir``."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "train": write_jsonl(directory / "train.jsonl", result.train),
        "val": write_jsonl(directory / "val.jsonl", result.val),
        "rejected": write_rejected_csv(directory / "rejected.csv", result.rejected),
    }
