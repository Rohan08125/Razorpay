"""Evidence-first controller interface for finance exception investigation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from investigate_exception import ExceptionInvestigator


def confidence_for(root_cause: str, evidence_count: int) -> float:
    base = {
        "MISSING_BANK_CREDIT": 0.99,
        "DUPLICATE_OR_CONFLICTING_BANK_CREDIT": 0.98,
        "BANK_AMOUNT_VARIANCE": 0.95,
        "SETTLEMENT_ITEM_COUNT_MISMATCH": 0.96,
        "SETTLEMENT_TOTAL_MISMATCH": 0.92,
        "NONE": 0.35,
    }.get(root_cause, 0.50)
    return round(min(0.99, base + min(evidence_count, 3) * 0.01), 2)


def build_case(settlement_id: str, investigator: ExceptionInvestigator) -> dict:
    finding = investigator.investigate_settlement(settlement_id)
    if "error" in finding:
        return {
            "case_id": settlement_id,
            "root_cause": "DATA_ERROR",
            "confidence": 0.0,
            "financial_impact": 0.0,
            "evidence": [finding["error"]],
            "recommended_action": "ESCALATE_FOR_MANUAL_REVIEW",
            "explainable": True,
        }

    evidence = finding.get("evidence", [])
    root_cause = finding["root_cause"]
    confidence = confidence_for(root_cause, len(evidence))
    financial_impact = 0.0

    for text in evidence:
        if "variance ₹" in text:
            try:
                financial_impact = abs(float(text.split("variance ₹", 1)[1]))
            except ValueError:
                financial_impact = 0.0
            break

    recommendations = finding.get("recommendations", ["Escalate for manual review"])
    recommendation = recommendations[0] if recommendations else "Escalate for manual review"
    if root_cause == "NONE":
        recommendation = "AUTO_RECONCILE"

    return {
        "case_id": settlement_id,
        "root_cause": root_cause,
        "confidence": confidence,
        "financial_impact": round(financial_impact, 2),
        "evidence": evidence,
        "recommended_action": recommendation,
        "explainable": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AI-ready investigation case packets")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    investigator = ExceptionInvestigator(args.data_dir)
    settlement_ids = list(investigator.data.settlement_by_id)
    cases = [build_case(settlement_id, investigator) for settlement_id in settlement_ids]

    output = args.results_dir / "ai_case_packets.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"Built {len(cases)} AI-ready case packets -> {output}")


if __name__ == "__main__":
    main()
