"""Cross-section coherence and consistency evaluation for generated reports.

Evaluates the logical flow and factual alignment of the full report from abstract
through analysis sections to conclusion, using only deterministic checks (no LLM
judge) so results stay reproducible.

Metrics:

* term_overlap_coherence — Jaccard overlap of domain concept terms between abstract
  and conclusion; measures thematic continuity from opening to closing.
* section_flow_coherence — fraction of analysis sections whose key terms appear in
  the conclusion; measures whether all findings feed into the final synthesis.
* numeric_consistency — fraction of numbers in abstract+conclusion that also appear
  in the analysis body; penalises claims unsupported by the data sections.
* dataset_grounding_consistency — fraction of ground-truth numeric facts that
  appear in at least one analysis section AND in the abstract/conclusion wrapper.
* claim_factual_consistency — fraction of fact-carrying sentences in the abstract
  whose numbers trace back to dataset-derived values in the analysis body.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from local_agentic_analytics.core.config import PROJECT_ROOT
from local_agentic_analytics.finetune.concept_bank import count_concept_terms
from local_agentic_analytics.finetune.validators import extract_numbers

_TRANSITION_TERMS: set[str] = {
    "oleh karena itu", "dengan demikian", "secara keseluruhan",
    "sebagai kesimpulan", "dari hasil analisis", "berdasarkan temuan",
    "hal ini menunjukkan", "hal ini mengindikasikan", "temuan ini",
    "pola ini", "hasil ini", "analisis ini",
}

_KNOWN_CHART_PROPER_NAMES: list[str] = [
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "Global_active_power", "Global_reactive_power",
    "Global_intensity", "Voltage",
]

DEFAULT_LATEX_PATH = PROJECT_ROOT / "reports" / "latex" / "energy_analysis_report.tex"
DEFAULT_GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "references" / "gold_reports" / "energy_report_ground_truth.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "report_coherence_eval.json"
)
DEFAULT_CSV_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "report_coherence_eval.csv"
)

_ABSTRACT_START_RE = re.compile(r"\\begin\{abstract\}")
_ABSTRACT_END_RE = re.compile(r"\\end\{abstract\}")
_SECTION_CMD_RE = re.compile(r"\\(?:section|subsection)\*?\{([^{}]*)\}")
_DETAILED_ANALYSIS_ID = "detailed analysis"
_SYNTHESIS_ID = "synthesis and implications"
_CONCLUSION_ID = "conclusion"
_FIGURE_RE = re.compile(r"\\includegraphics.*\n")


def _extract_abstract(latex_text: str) -> str:
    start = _ABSTRACT_START_RE.search(latex_text)
    end = _ABSTRACT_END_RE.search(latex_text)
    if start and end and start.end() < end.start():
        return latex_text[start.end():end.start()]
    return ""


def _extract_section(latex_text: str, section_name: str) -> str:
    norm = section_name.strip().lower()
    escaped_name = re.escape(norm).replace(r"\ ", r"\s+")
    pattern = (
        r"\\(?:section|subsection)\*?\{"
        r"[^{}]*"
        + escaped_name +
        r"[^{}]*"
        r"\}"
    )
    match = re.search(pattern, latex_text, re.IGNORECASE)
    if not match:
        return ""
    start_pos = match.end()
    rest = latex_text[start_pos:]
    next_sec = _SECTION_CMD_RE.search(rest)
    if next_sec:
        return rest[:next_sec.start()]
    end_doc = rest.find(r"\end{document}")
    if end_doc >= 0:
        return rest[:end_doc]
    return rest


def _extract_analysis_sections(latex_text: str) -> dict[str, str]:
    find_section_pattern = re.compile(r"\\(?:section|subsection)\*?\{([^{}]*)\}", re.IGNORECASE)
    detailed_start = None
    detailed_end = None

    for m in re.finditer(r"\\(?:section)\*?\{([^{}]*)\}", latex_text, re.IGNORECASE):
        title = m.group(1).strip()
        if _DETAILED_ANALYSIS_ID in title.lower():
            detailed_start = m.end()
        elif detailed_start is not None and detailed_end is None:
            detailed_end = m.start()

    if detailed_start is None:
        return {}
    if detailed_end is None:
        end_marker = latex_text.find(r"\end{document}")
        detailed_end = end_marker if end_marker >= 0 else len(latex_text)

    detailed_range = latex_text[detailed_start:detailed_end]

    subsections: dict[str, str] = {}
    sub_matches = list(re.finditer(r"\\(?:subsection)\*?\{([^{}]*)\}", detailed_range, re.IGNORECASE))
    for i, m in enumerate(sub_matches):
        title = m.group(1).strip()
        body_start = m.end()
        body_end = sub_matches[i + 1].start() if i + 1 < len(sub_matches) else len(detailed_range)
        body = detailed_range[body_start:body_end]
        subsections[title] = body

    return subsections


def _latex_to_plain_text(latex_text: str) -> str:
    text = latex_text.replace(r"\_", "_")
    text = text.replace(r"\%", "%")
    text = re.sub(r"\\(?:section|subsection|caption|title|author)\*?\{([^{}]*)\}", r" \1 ", text)
    text = re.sub(r"\\(?:begin|end)\{[^{}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z]+(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", text)
    text = re.sub(r"[{}$]", " ", text)
    return " ".join(text.split())


def _domain_terms(text: str) -> set[str]:
    tokens: set[str] = set()
    text_lower = text.lower()
    for term in _KNOWN_CHART_PROPER_NAMES:
        if term.lower() in text_lower:
            tokens.add(term.lower())
    return tokens


def _analysis_numbers(text: str) -> set[float]:
    numbers: set[float] = set()
    for match in re.finditer(r"(?<![\w.,])[-−]?\d+(?:[.,]\d+)*(?=\W|$)", text):
        try:
            val = float(match.group().replace(",", ".").replace("−", "-"))
            if val > 0.001:
                numbers.add(round(val, 4))
        except ValueError:
            pass
    return numbers


def _factual_sentences(text: str) -> list[str]:
    raw = text.split(". ")
    return [s.strip() + "." for s in raw if s.strip() and re.search(r"\d", s)]


def evaluate_term_overlap_coherence(abstract: str, conclusion: str) -> dict[str, Any]:
    abs_terms = _domain_terms(abstract)
    con_terms = _domain_terms(conclusion)
    union = abs_terms | con_terms
    if not union:
        return {"score": 1.0, "abstract_terms": [], "conclusion_terms": [], "shared": []}
    shared = abs_terms & con_terms
    return {
        "score": round(len(shared) / len(union), 4),
        "abstract_terms": sorted(abs_terms),
        "conclusion_terms": sorted(con_terms),
        "shared": sorted(shared),
    }


def evaluate_section_flow_coherence(
    analysis_sections: dict[str, str],
    conclusion: str,
) -> dict[str, Any]:
    if not analysis_sections or not conclusion:
        return {"score": 1.0, "section_links": [], "linked_count": 0, "total_sections": 0}

    con_lower = conclusion.lower()
    section_links: list[dict[str, Any]] = []
    for title, body in analysis_sections.items():
        body_terms = _domain_terms(body)
        linked = bool(body_terms and any(t in con_lower for t in body_terms))
        if not linked:
            body_nums = _analysis_numbers(body)
            con_nums = _analysis_numbers(conclusion)
            shared_nums = body_nums & con_nums
            linked = len(shared_nums) >= 1
        section_links.append({"section_title": title, "linked_to_conclusion": linked})

    linked_count = sum(1 for sl in section_links if sl["linked_to_conclusion"])
    return {
        "score": round(linked_count / len(section_links), 4) if section_links else 1.0,
        "section_links": section_links,
        "linked_count": linked_count,
        "total_sections": len(section_links),
    }


def _numbers_overlap(
    wrapper_nums: set[float],
    analysis_nums: set[float],
    rel_tol: float = 0.02,
) -> tuple[list[float], list[float]]:
    grounded: list[float] = []
    ungrounded: list[float] = []
    for wn in wrapper_nums:
        matched = False
        for an in analysis_nums:
            if abs(an) > 0:
                if abs(wn - an) / abs(an) <= rel_tol:
                    matched = True
                    break
            elif abs(wn - an) <= 0.01:
                matched = True
                break
        if matched:
            grounded.append(round(wn, 4))
        else:
            ungrounded.append(round(wn, 4))
    return grounded, ungrounded


def evaluate_numeric_consistency(
    abstract: str,
    conclusion: str,
    analysis_text: str,
) -> dict[str, Any]:
    abs_nums = _analysis_numbers(abstract) if abstract else set()
    con_nums = _analysis_numbers(conclusion) if conclusion else set()
    wrapper_nums = abs_nums | con_nums
    analysis_nums = _analysis_numbers(analysis_text) if analysis_text else set()

    if not wrapper_nums:
        return {"score": 1.0, "wrapper_total": 0, "grounded": 0, "ungrounded": []}

    grounded, ungrounded = _numbers_overlap(wrapper_nums, analysis_nums)

    return {
        "score": round(len(grounded) / len(wrapper_nums), 4),
        "wrapper_total": len(wrapper_nums),
        "grounded": len(grounded),
        "ungrounded": ungrounded,
    }


def evaluate_dataset_grounding_consistency(
    latex_text: str,
    ground_truth_path: str | Path = DEFAULT_GROUND_TRUTH_PATH,
) -> dict[str, Any]:
    gt_path = Path(ground_truth_path)
    if not gt_path.is_file():
        return {"score": 1.0, "facts": [], "note": "ground truth file not found"}

    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    numeric_facts = gt.get("numeric_facts", [])
    if not isinstance(numeric_facts, list) or not numeric_facts:
        return {"score": 1.0, "facts": [], "note": "no numeric facts in ground truth"}

    abstract = _extract_abstract(latex_text)
    conclusion = _extract_section(latex_text, "Conclusion")
    analysis_sections = _extract_analysis_sections(latex_text)
    analysis_body = " ".join(analysis_sections.values())

    wrapper_text = " ".join([abstract, conclusion])
    plain_wrapper = _latex_to_plain_text(wrapper_text)
    plain_analysis = _latex_to_plain_text(analysis_body)

    fact_results: list[dict[str, Any]] = []
    for fact in numeric_facts:
        fid = fact.get("id", "")
        value = fact.get("value")
        tolerance = float(fact.get("tolerance", 0.0001))
        unit = str(fact.get("unit", ""))
        variants = _numeric_value_variants_simple(value) if value is not None else []

        in_analysis = any(v in plain_analysis for v in variants)
        in_wrapper = any(v in plain_wrapper for v in variants)

        fact_results.append({
            "id": fid,
            "value": value,
            "in_analysis": in_analysis,
            "in_wrapper": in_wrapper,
            "flow_through": in_analysis and in_wrapper,
        })

    flow_count = sum(1 for fr in fact_results if fr["flow_through"])
    return {
        "score": round(flow_count / len(fact_results), 4) if fact_results else 1.0,
        "facts": fact_results,
        "flow_through_count": flow_count,
        "total_facts": len(fact_results),
    }


def _numeric_value_variants_simple(value) -> list[str]:
    if value is None:
        return []
    try:
        v = float(value)
    except (TypeError, ValueError):
        return [str(value)]
    variants = []
    for decimals in range(2, 7):
        formatted = f"{v:.{decimals}f}"
        trimmed = formatted.rstrip("0").rstrip(".")
        for c in (formatted, trimmed):
            if c not in variants:
                variants.append(c)
    if float(v).is_integer():
        iv = str(int(v))
        if iv not in variants:
            variants.append(iv)
    return variants


def evaluate_claim_factual_consistency(
    abstract: str,
    analysis_text: str,
) -> dict[str, Any]:
    if not abstract or not analysis_text:
        return {"score": 1.0, "total_claims": 0, "grounded_claims": 0, "claims": []}

    abs_claims = _factual_sentences(abstract)
    if not abs_claims:
        return {"score": 1.0, "total_claims": 0, "grounded_claims": 0, "claims": []}

    analysis_nums = _analysis_numbers(analysis_text)
    claim_results: list[dict[str, Any]] = []
    for claim in abs_claims:
        claim_nums = _analysis_numbers(claim)
        grounded = bool(claim_nums and all(n in analysis_nums for n in claim_nums))
        claim_results.append({
            "claim": claim[:200],
            "numbers_in_claim": [round(n, 4) for n in claim_nums],
            "grounded": grounded,
        })

    grounded_count = sum(1 for c in claim_results if c["grounded"])
    return {
        "score": round(grounded_count / len(claim_results), 4) if claim_results else 1.0,
        "total_claims": len(claim_results),
        "grounded_claims": grounded_count,
        "claims": claim_results,
    }


def evaluate_report_coherence(
    latex_path: str | Path = DEFAULT_LATEX_PATH,
    ground_truth_path: str | Path = DEFAULT_GROUND_TRUTH_PATH,
) -> dict[str, Any]:
    latex_file = Path(latex_path)
    if not latex_file.is_file():
        raise FileNotFoundError(f"LaTeX file not found: {latex_file}")
    latex_text = latex_file.read_text(encoding="utf-8")

    abstract = _extract_abstract(latex_text)
    conclusion = _extract_section(latex_text, "Conclusion")
    analysis_sections = _extract_analysis_sections(latex_text)
    analysis_body = " ".join(analysis_sections.values())

    plain_abstract = _latex_to_plain_text(abstract)
    plain_conclusion = _latex_to_plain_text(conclusion)
    plain_analysis = _latex_to_plain_text(analysis_body)

    term_overlap = evaluate_term_overlap_coherence(plain_abstract, plain_conclusion)
    section_flow = evaluate_section_flow_coherence(analysis_sections, plain_conclusion)
    numeric_consistency = evaluate_numeric_consistency(
        plain_abstract, plain_conclusion, plain_analysis
    )
    dataset_grounding = evaluate_dataset_grounding_consistency(
        latex_text, ground_truth_path
    )
    claim_consistency = evaluate_claim_factual_consistency(
        plain_abstract, plain_analysis
    )

    coherence_score = round((term_overlap["score"] + section_flow["score"]) / 2.0, 4)
    consistency_score = round(
        (numeric_consistency["score"] + dataset_grounding["score"] + claim_consistency["score"]) / 3.0,
        4,
    )
    narrative_quality = round((coherence_score + consistency_score) / 2.0, 4)

    return {
        "coherence_metrics": {
            "term_overlap": term_overlap,
            "section_flow": section_flow,
            "composite_coherence": coherence_score,
        },
        "consistency_metrics": {
            "numeric_consistency": numeric_consistency,
            "dataset_grounding": dataset_grounding,
            "claim_factual": claim_consistency,
            "composite_consistency": consistency_score,
        },
        "composite_narrative_quality": narrative_quality,
        "abstract_length_chars": len(plain_abstract),
        "conclusion_length_chars": len(plain_conclusion),
        "analysis_sections_count": len(analysis_sections),
        "analysis_section_titles": sorted(analysis_sections.keys()),
    }


def write_coherence_result(
    result: dict[str, Any],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    csv_path: str | Path = DEFAULT_CSV_PATH,
) -> dict[str, Path]:
    json_out = Path(output_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_out = Path(csv_path)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "category": "coherence",
            "metric": "term_overlap",
            "score": result["coherence_metrics"]["term_overlap"]["score"],
        },
        {
            "category": "coherence",
            "metric": "section_flow",
            "score": result["coherence_metrics"]["section_flow"]["score"],
        },
        {
            "category": "coherence",
            "metric": "composite_coherence",
            "score": result["coherence_metrics"]["composite_coherence"],
        },
        {
            "category": "consistency",
            "metric": "numeric_consistency",
            "score": result["consistency_metrics"]["numeric_consistency"]["score"],
        },
        {
            "category": "consistency",
            "metric": "dataset_grounding",
            "score": result["consistency_metrics"]["dataset_grounding"]["score"],
        },
        {
            "category": "consistency",
            "metric": "claim_factual",
            "score": result["consistency_metrics"]["claim_factual"]["score"],
        },
        {
            "category": "consistency",
            "metric": "composite_consistency",
            "score": result["consistency_metrics"]["composite_consistency"],
        },
        {
            "category": "overall",
            "metric": "narrative_quality",
            "score": result["composite_narrative_quality"],
        },
    ]
    with csv_out.open("w", encoding="utf-8", newline="") as fh:
        import csv
        w = csv.DictWriter(fh, fieldnames=["category", "metric", "score"])
        w.writeheader()
        w.writerows(rows)

    return {"json": json_out, "csv": csv_out}
