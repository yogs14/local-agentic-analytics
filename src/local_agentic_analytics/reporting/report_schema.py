"""Dataclasses for analysis report generation."""

from __future__ import annotations

from dataclasses import dataclass, field


def _validate_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


@dataclass
class ReportFigure:
    figure_id: str
    title: str
    path: str
    caption: str

    def __post_init__(self) -> None:
        _validate_non_empty(self.figure_id, "figure_id")
        _validate_non_empty(self.title, "title")
        _validate_non_empty(self.path, "path")
        _validate_non_empty(self.caption, "caption")


@dataclass
class ReportSection:
    title: str
    content: str
    figures: list[ReportFigure] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_non_empty(self.title, "title")
        _validate_non_empty(self.content, "content")


@dataclass
class AnalysisReport:
    title: str
    author: str
    abstract: str
    introduction: str
    methodology: str
    sections: list[ReportSection]
    synthesis: str
    conclusion: str

    def __post_init__(self) -> None:
        _validate_non_empty(self.title, "title")
        _validate_non_empty(self.author, "author")
        _validate_non_empty(self.abstract, "abstract")
        _validate_non_empty(self.introduction, "introduction")
        _validate_non_empty(self.methodology, "methodology")
        _validate_non_empty(self.synthesis, "synthesis")
        _validate_non_empty(self.conclusion, "conclusion")
        if not isinstance(self.sections, list):
            raise ValueError("sections must be a list")
