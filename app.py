"""Streamlit dashboard for the local financial reconciliation system."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from ai_controller import build_case
from investigate_exception import ExceptionInvestigator
from ollama_agent import investigate


st.set_page_config(
    page_title="Financial Reconciliation AI",
    page_icon="💳",
    layout="wide",
)


@st.cache_data
def read_csv(name: str) -> list[dict[str, str]]:
    path = RESULTS_DIR / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


@st.cache_data
def load_json(name: str, default: Any) -> Any:
    path = RESULTS_DIR / name
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def money(value: Any) -> str:
    try:
        return f"₹{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


settlements = read_csv("settlement_reconciliation.csv")
bank_rows = read_csv("bank_reconciliation.csv")
ollama_summary = load_json("ollama_agent_summary.json", {})
ollama_decisions = load_json("ollama_agent_decisions.json", [])
ollama_evaluation = load_json("ollama_evaluation.json", {})

st.title("💳 Financial Reconciliation AI")
st.caption("Deterministic reconciliation + local Qwen 0.6B evidence explanation")

if not settlements:
    st.error("Settlement reconciliation results are missing. Run the reconciliation pipeline first.")
    st.stop()

# ---- KPI row ---------------------------------------------------------------
total = len(settlements)
exceptions = sum(r.get("status") == "EXCEPTION" for r in settlements)
reconciled = total - exceptions
recon_rate = (reconciled / total) if total else 0

bank_total = len(bank_rows)
bank_exceptions = sum(r.get("status") == "EXCEPTION" for r in bank_rows)
bank_rate = ((bank_total - bank_exceptions) / bank_total) if bank_total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Settlements", f"{total:,}")
c2.metric("Exceptions", f"{exceptions:,}")
c3.metric("Settlement Reconciliation", f"{recon_rate:.1%}")
c4.metric("Bank Reconciliation", f"{bank_rate:.1%}")

st.divider()

# ---- Exception explorer ---------------------------------------------------
st.subheader("Exception Explorer")

exception_rows = [r for r in settlements if r.get("status") == "EXCEPTION"]
ids = [r["settlement_id"] for r in exception_rows]

selected_id = st.selectbox(
    "Select a settlement",
    ids,
    index=0 if ids else None,
)

selected = next((r for r in exception_rows if r["settlement_id"] == selected_id), None)

if selected:
    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Reconciliation Finding")
        st.write(f"**Settlement:** `{selected['settlement_id']}`")
        st.write(f"**Merchant:** `{selected.get('merchant_id', '—')}`")
        st.write(f"**Status:** `{selected['status']}`")
        st.write(f"**Items found:** {selected.get('item_count', '—')}")
        st.write(f"**Expected item net:** {money(selected.get('expected_item_net'))}")
        st.write(f"**Recorded net:** {money(selected.get('recorded_net'))}")

        issues = selected.get("issues", "")
        st.markdown("**Observable issues**")
        if issues:
            for issue in issues.split("|"):
                st.warning(issue)
        else:
            st.info("No issue codes recorded.")

    with right:
        st.markdown("### AI Investigation")

        existing = next(
            (x for x in ollama_decisions if x.get("case_id") == selected_id),
            None,
        )

        if st.button("Run local Qwen investigation", type="primary"):
            with st.spinner("Investigating locally with Qwen 0.6B..."):
                try:
                    investigator = ExceptionInvestigator(DATA_DIR)
                    case = build_case(selected_id, investigator)
                    decision = investigate(case, str(DATA_DIR))
                    decision["case_id"] = selected_id
                    st.session_state[f"decision_{selected_id}"] = decision
                except Exception as exc:
                    st.error(f"Investigation failed: {exc}")
                    decision = None
        else:
            decision = st.session_state.get(f"decision_{selected_id}")

        if decision is None and existing:
            decision = existing
            st.info(
                "Showing cached Qwen decision. "
                "Click the button above to run a fresh investigation."
            )

        if decision:
            d1, d2 = st.columns(2)
            d1.metric("Confidence", f"{float(decision.get('confidence', 0)):.0%}")
            d2.metric("Tool Calls", str(decision.get("tool_calls", 0)))

            st.markdown(f"**Root cause:** `{decision.get('root_cause', '—')}`")
            st.markdown(f"**Action:** `{decision.get('action', '—')}`")

            st.markdown("**AI rationale**")
            st.info(decision.get("rationale") or "No rationale returned.")

            evidence_used = decision.get("evidence_used") or []
            if evidence_used:
                st.markdown("**Evidence used**")
                for evidence in evidence_used:
                    st.write(f"• {evidence}")

# ---- Benchmark ------------------------------------------------------------
st.divider()
st.subheader("Benchmark")

if ollama_evaluation.get("ground_truth_cases"):
    st.markdown("#### Ground-truth reconciliation benchmark")
    st.caption(
        "Measures the deterministic reconciliation engine against the controlled "
        "corruption recorded in data/ground_truth.csv."
    )

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Corruption recall", f"{ollama_evaluation.get('detection_recall', 0):.1%}")
    g2.metric("Detection precision", f"{ollama_evaluation.get('detection_precision', 0):.1%}")
    g3.metric("Detection F1", f"{ollama_evaluation.get('detection_f1', 0):.1%}")
    g4.metric("Root-cause accuracy", f"{ollama_evaluation.get('ground_truth_root_cause_accuracy', 0):.1%}")

    st.write(
        f"Detected **{ollama_evaluation.get('corrupted_cases_detected', 0)}** of "
        f"{ollama_evaluation.get('ground_truth_cases', 0)} corrupted settlements; "
        f"missed **{ollama_evaluation.get('missed_corrupted_cases', 0)}** and produced "
        f"**{ollama_evaluation.get('false_positive_cases', 0)}** false positives."
    )

    type_rows = []
    for corruption_type, metrics in (ollama_evaluation.get("by_corruption_type") or {}).items():
        type_rows.append({
            "Corruption type": corruption_type,
            "Cases": metrics.get("cases", 0),
            "Root-cause accuracy": f"{metrics.get('root_cause_accuracy', 0):.1%}",
        })
    if type_rows:
        st.dataframe(type_rows, width="stretch", hide_index=True)
else:
    st.warning(
        "Ground-truth benchmark is unavailable. Run `py src\\evaluate_ollama.py` "
        "after restoring the local data/ground_truth.csv dataset."
    )

st.markdown("#### Local Qwen consistency benchmark")
b1, b2, b3, b4 = st.columns(4)
b1.metric("Cases evaluated", str(ollama_evaluation.get("ollama_compared_cases", ollama_summary.get("requested_cases", len(ollama_decisions)))))
b2.metric("Root-cause agreement", f"{ollama_evaluation.get('ollama_root_cause_accuracy', 0):.1%}" if ollama_evaluation else "—")
b3.metric("Action agreement", f"{ollama_evaluation.get('ollama_action_accuracy', 0):.1%}" if ollama_evaluation else "—")
b4.metric("Fallback rate", f"{ollama_evaluation.get('ollama_fallback_rate', 0):.1%}" if ollama_evaluation else "—")

st.caption(
    "The deterministic reconciliation engine remains the financial source of truth. "
    "Qwen is used as an evidence-based explanation layer and cannot override the decision."
)

# ---- Settlement table -----------------------------------------------------
st.divider()
st.subheader("Settlement Overview")

filter_status = st.radio(
    "Show",
    ["All", "Exceptions", "Reconciled"],
    horizontal=True,
)

if filter_status == "Exceptions":
    display_rows = exception_rows
elif filter_status == "Reconciled":
    display_rows = [r for r in settlements if r.get("status") == "RECONCILED"]
else:
    display_rows = settlements

st.dataframe(
    display_rows,
    width="stretch",
    hide_index=True,
)
