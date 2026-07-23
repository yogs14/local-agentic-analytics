from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.report_coherence_eval import (
    _extract_abstract,
    _extract_section,
    _extract_analysis_sections,
    _analysis_numbers,
    _domain_terms,
    _latex_to_plain_text,
    _factual_sentences,
    evaluate_term_overlap_coherence,
    evaluate_section_flow_coherence,
    evaluate_numeric_consistency,
    evaluate_dataset_grounding_consistency,
    evaluate_claim_factual_consistency,
    evaluate_report_coherence,
    write_coherence_result,
)


_SIMPLE_LATEX = r"""
\documentclass{article}
\begin{document}
\begin{abstract}
Laporan ini menganalisis konsumsi daya listrik rumah tangga. Rata-rata Global_active_power tercatat 1,09 kW.
\end{abstract}

\section{Introduction}
Pengantar singkat.

\section{Detailed Analysis}
\subsection{Tren Harian}
Rata-rata Global_active_power harian 1,092 kW dengan rentang 0,17 kW hingga 3,31 kW.
\begin{figure}[H]
\includegraphics{../figures/daily_active_power_trend.png}
\end{figure}

\subsection{Pola Per Jam}
Beban puncak 1,90 kW pada pukul 20.00 dan beban dasar 0,44 kW pada pukul 04.00. Voltage stabil pada 240,84 V.

\subsection{Distribusi Daya}
Distribusi Global_active_power dengan modus bin 0,2 kW, rata-rata 1,0916 kW, rentang 0,076-11,122 kW.

\subsection{Distribusi Tegangan}
Rata-rata Voltage 240,8399 V, simpangan baku 3,24 V, rentang 223,2-254,15 V.

\subsection{Korelasi}
Korelasi Global_active_power vs Global_intensity sebesar 0,9989 menunjukkan hubungan sangat kuat.

\subsection{Sub Metering}
Sub_metering_3 mendominasi dengan rata-rata 6,4584 Wh.

\section{Conclusion}
Konsumsi daya rumah tangga menunjukkan disparitas puncak-lembah signifikan. Global_active_power rata-rata 1,09 kW dengan Sub_metering_3 sebagai kontributor terbesar. Strategi demand response dapat diterapkan.
\end{document}
"""


def test_extract_abstract():
    abstract = _extract_abstract(_SIMPLE_LATEX)
    assert "Global_active_power" in abstract


def test_extract_abstract_empty_when_missing():
    assert _extract_abstract("\\begin{document}\\end{document}") == ""


def test_extract_section():
    conclusion = _extract_section(_SIMPLE_LATEX, "Conclusion")
    assert "demand response" in conclusion


def test_extract_section_empty_when_missing():
    assert _extract_section(_SIMPLE_LATEX, "NonexistentSection") == ""


def test_extract_analysis_sections():
    sections = _extract_analysis_sections(_SIMPLE_LATEX)
    assert len(sections) >= 4
    assert any("Tren" in t for t in sections)


def test_analysis_numbers():
    nums = _analysis_numbers("rata-rata 1,092 kW, rentang 0,17 kW hingga 3,31 kW")
    assert 1.092 in nums or round(1.092, 4) in nums


def test_domain_terms():
    terms = _domain_terms("Global_active_power dan Sub_metering_3 adalah metrik utama")
    assert "global_active_power" in terms
    assert "sub_metering_3" in terms


def test_latex_to_plain_text():
    plain = _latex_to_plain_text(
        r"\section{Abstract}\n Rata-rata Global\_active\_power 1,092 kW."
    )
    assert "Global_active_power" in plain
    assert "1,092" in plain


def test_factual_sentences():
    sentences = _factual_sentences("Normal text. Rata-rata 1,09 kW. Kesimpulan.")
    assert len(sentences) >= 1


def test_term_overlap_coherence():
    abstract = "Global_active_power rata-rata 1,09 kW dan Voltage stabil."
    conclusion = "Global_active_power menunjukkan disparitas. Voltage pada 240 V."
    result = evaluate_term_overlap_coherence(abstract, conclusion)
    assert result["score"] >= 0.0
    assert len(result["shared"]) >= 1


def test_term_overlap_no_shared():
    result = evaluate_term_overlap_coherence(
        "Global_active_power 1,09 kW", "Sub_metering_3 6,46 Wh"
    )
    assert result["score"] == 0.0


def test_section_flow_coherence():
    sections = {
        "Tren Harian": "Global_active_power 1,092 kW",
        "Sub Metering": "Sub_metering_3 6,4584 Wh",
    }
    conclusion = "Sub_metering_3 mendominasi konsumsi rumah tangga."
    result = evaluate_section_flow_coherence(sections, conclusion)
    assert result["score"] >= 0.0
    linked = {sl["section_title"]: sl["linked_to_conclusion"] for sl in result["section_links"]}
    assert linked.get("Sub Metering") is True


def test_numeric_consistency_all_grounded():
    result = evaluate_numeric_consistency(
        abstract="Rata-rata 240,84 V.",
        conclusion="Tegangan 240,84 V stabil.",
        analysis_text="Rata-rata Voltage 240,8399 V dengan simpangan baku 3,24 V.",
    )
    assert result["wrapper_total"] > 0
    assert result["ungrounded"] == []


def test_numeric_consistency_ungrounded_detected():
    result = evaluate_numeric_consistency(
        abstract="Konsumsi 999,9 kW.",
        conclusion="",
        analysis_text="Rata-rata daya 1,092 kW.",
    )
    assert len(result["ungrounded"]) >= 1


def test_dataset_grounding_consistency(tmp_path):
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text("""{
  "numeric_facts": [
    {"id": "N001", "value": 1.092, "unit": "kW", "tolerance": 0.01}
  ]
}""")
    result = evaluate_dataset_grounding_consistency(
        _SIMPLE_LATEX, gt_path
    )
    assert result["total_facts"] >= 1


def test_claim_factual_consistency():
    result = evaluate_claim_factual_consistency(
        abstract="Rata-rata daya 1,09 kW dengan beban puncak 1,90 kW.",
        analysis_text="Rata-rata Global_active_power harian 1,092 kW dengan beban puncak 1,899064 kW.",
    )
    assert result["grounded_claims"] >= 0


def test_evaluate_report_coherence_integration(tmp_path):
    latex_path = tmp_path / "test.tex"
    latex_path.write_text(_SIMPLE_LATEX)

    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text("""{
  "numeric_facts": [
    {"id": "N001", "value": 1.092, "unit": "kW", "tolerance": 0.01}
  ]
}""")

    result = evaluate_report_coherence(latex_path, gt_path)
    assert "coherence_metrics" in result
    assert "consistency_metrics" in result
    assert "composite_narrative_quality" in result
    assert 0.0 <= result["composite_narrative_quality"] <= 1.0


def test_write_coherence_result(tmp_path):
    result = {
        "coherence_metrics": {
            "term_overlap": {"score": 0.75},
            "section_flow": {"score": 0.83},
            "composite_coherence": 0.79,
        },
        "consistency_metrics": {
            "numeric_consistency": {"score": 0.9},
            "dataset_grounding": {"score": 0.5},
            "claim_factual": {"score": 0.8},
            "composite_consistency": 0.73,
        },
        "composite_narrative_quality": 0.76,
    }
    paths = write_coherence_result(
        result,
        output_path=tmp_path / "coherence.json",
        csv_path=tmp_path / "coherence.csv",
    )
    assert paths["json"].is_file()
    assert paths["csv"].is_file()
