"""Render LaTeX reports from templates."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

from jinja2 import Environment, FileSystemLoader

from local_agentic_analytics.reporting.report_schema import AnalysisReport


LATEX_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "%": r"\%",
    "&": r"\&",
    "_": r"\_",
    "#": r"\#",
    "$": r"\$",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(value: Any) -> str:
    text = "" if value is None else str(value)
    return "".join(LATEX_ESCAPE_MAP.get(char, char) for char in text)


def normalize_figure_path(path: str | Path, base_dir: str | Path) -> str:
    figure_path = Path(path)
    base_path = Path(base_dir)

    try:
        if figure_path.is_absolute():
            normalized = os.path.relpath(figure_path, start=base_path)
        else:
            normalized = str(figure_path)
    except ValueError:
        normalized = str(figure_path)

    return normalized.replace("\\", "/")


def latex_label(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    label = re.sub(r"[^A-Za-z0-9:-]+", "-", text)
    return label.strip("-") or "figure"


def render_latex_report(
    report: AnalysisReport,
    template_path: str | Path,
    output_path: str | Path,
) -> Path:
    template_file = Path(template_path)
    if not template_file.exists():
        raise FileNotFoundError(f"LaTeX template not found: {template_file}")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    environment = Environment(
        loader=FileSystemLoader(str(template_file.parent)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters["latex_escape"] = escape_latex
    environment.filters["latex_label"] = latex_label
    environment.filters["latex_path"] = (
        lambda value: normalize_figure_path(value, output_file.parent)
    )

    template = environment.get_template(template_file.name)
    rendered = template.render(report=report)
    output_file.write_text(rendered, encoding="utf-8")
    return output_file
