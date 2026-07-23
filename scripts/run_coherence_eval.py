from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.report_coherence_eval import (
    evaluate_report_coherence,
    write_coherence_result,
)


def main() -> int:
    result = evaluate_report_coherence()
    paths = write_coherence_result(result)

    cm = result["coherence_metrics"]
    cs = result["consistency_metrics"]

    print(f"Coherence & Consistency evaluation saved to: {paths['json']}")
    print(f"CSV summary: {paths['csv']}")
    print()
    print("-- Coherence --")
    print(f"  term_overlap (abstract<->conclusion): {cm['term_overlap']['score']:.4f}")
    print(f"  section_flow (analysis->conclusion):  {cm['section_flow']['score']:.4f}")
    print(f"  composite_coherence:                  {cm['composite_coherence']:.4f}")
    print()
    print("-- Consistency --")
    print(f"  numeric_consistency (wrapper vs body):{cs['numeric_consistency']['score']:.4f}")
    print(f"  dataset_grounding (end-to-end flow):  {cs['dataset_grounding']['score']:.4f}")
    print(f"  claim_factual (abstract grounding):   {cs['claim_factual']['score']:.4f}")
    print(f"  composite_consistency:                {cs['composite_consistency']:.4f}")
    print()
    print(f"-- Composite Narrative Quality: {result['composite_narrative_quality']:.4f} --")

    if cm["section_flow"]["score"] < 1.0:
        unlinked = [s["section_title"] for s in cm["section_flow"]["section_links"]
                     if not s["linked_to_conclusion"]]
        print(f"\nSections not linked to conclusion: {', '.join(unlinked)}")

    if cs["numeric_consistency"].get("ungrounded"):
        print(f"\nUngrounded numbers in abstract/conclusion: "
              f"{cs['numeric_consistency']['ungrounded']}")

    dg = cs["dataset_grounding"]
    if dg.get("facts"):
        missing = [f"{f['id']}({f['value']})" for f in dg["facts"]
                    if not f["flow_through"]]
        if missing:
            print(f"\nFacts missing flow-through: {', '.join(missing)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
