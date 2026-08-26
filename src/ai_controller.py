"""Evidence-first controller interface for finance exception investigation.

The finance rules remain deterministic. An LLM can consume the structured case
packet produced here to explain, classify, and recommend an action without
being allowed to alter financial records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from investigate_exception import load_data, investigate_settlement


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


def build_case(settlement_id: str, data: dict) -> dict:
    finding = investigate_settlement(settlement_id, data)
    confidence = confidence_for(finding["root_cause"], len(finding["evidence"]))
    recommendation = finding["recommendation"]

    if finding["root_cause"] == "NONE":
        recommendation = "ESCALATE_FOR_MANUAL_REVIEW"

    return {
        "case_id": settlement_id,
        "root_cause": finding["root_cause"],
        "confidence": confidence,
        "financial_impact": finding.get("financial_impact", 0.0),
        "evidence": finding["evidence"],
        "recommended_action": recommendation,
        "explainable": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AI-ready investigation case packets")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    data = load_data(args.data_dir)
    cases = []
    for settlement_id in data["settlements"]:
        cases.append(build_case(settlement_id, data))

    output = args.results_dir / "ai_case_packets.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(cases, file, indent=2)

    print(f"Built {len(cases)} AI-ready case packets -> {output}")


if __name__ == "__main__":
    main()
