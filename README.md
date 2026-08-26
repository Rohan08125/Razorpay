# Razorpay Financial Reconciliation AI

A local-first financial reconciliation system that combines deterministic accounting checks with a small local Qwen model for evidence-based exception explanations.

## Why this project

Financial reconciliation should be **correct before it is intelligent**. This project therefore separates the financial decision from the language model:

- The deterministic engine calculates reconciliation outcomes, root causes, financial impact, and recommended actions.
- A compact evidence packet is sent to local Ollama/Qwen for a human-readable explanation.
- The model cannot override the deterministic root cause or recommended action.
- Invalid model output falls back to a deterministic evidence explanation.

The core design principle is simple: **the LLM explains the financial decision; it does not make the financial decision.**

## Problem

Settlement operations need to reconcile:

- customer payments
- settlement items
- settlement batch totals
- bank credits

Failures are rarely a single obvious mismatch. Missing items, duplicate credits, fee drift, wrong references, amount corruption, and partial settlements can produce overlapping symptoms. A useful reconciliation system therefore needs reproducible rules, auditable evidence, and conservative financial actions.

## What it does

- Reconciles settlements, settlement items, payments, and bank transactions.
- Detects settlement and bank-side exceptions including:
  - missing or duplicate settlement items
  - settlement item amount mismatches
  - fee mismatches
  - missing or duplicate bank transactions
  - bank amount mismatches
  - wrong bank references
  - partial settlements
  - unexplained variance
- Builds compact evidence packets for local Qwen inference.
- Provides a Streamlit operations dashboard for exception investigation.
- Benchmarks the deterministic engine against controlled corruption with ground truth.
- Benchmarks Qwen for consistency with deterministic root causes and actions.

## Architecture

```text
Generated CSV data
        │
        ▼
Deterministic reconciliation + investigation
        │
        ├── root cause
        ├── financial impact
        └── recommended action
                 │
                 ▼
        Compact evidence packet
                 │
                 ▼
           Ollama / Qwen 0.6B
                 │
                 ▼
       rationale + evidence_used

Financial decision authority: deterministic engine
LLM authority: explanation only
```

## Deterministic engine

The deterministic layer is the financial source of truth for:

- exception detection
- root-cause classification
- financial impact
- recommended action

The settlement investigation validates each settlement item against the original payment gross amount. This catches controlled `AMOUNT_MISMATCH` corruption even when the item count and aggregate settlement net still appear internally consistent.

## Corruption benchmark

The controlled corruption suite contains 666 cases across:

- `MISSING_SETTLEMENT_ITEM`
- `DUPLICATE_SETTLEMENT_ITEM`
- `FEE_MISMATCH`
- `AMOUNT_MISMATCH`
- `MISSING_BANK_TRANSACTION`
- `DUPLICATE_BANK_TRANSACTION`
- `BANK_AMOUNT_MISMATCH`
- `WRONG_BANK_REFERENCE`
- `PARTIAL_SETTLEMENT`
- `UNEXPLAINED_VARIANCE`

The benchmark is designed so the deterministic engine can be evaluated against explicit ground truth rather than against model-generated labels.

## AI explanation layer

The local AI layer uses Ollama with `qwen3:0.6b`. The model is intentionally small so it can run locally on modest hardware.

The pipeline is:

1. Build a deterministic investigation case.
2. Construct a compact evidence packet rather than sending the full dataset.
3. Ask Qwen for strict JSON containing only a rationale and evidence list.
4. Retry once in generic JSON mode if structured output fails.
5. Fall back to deterministic evidence wording if the model still fails.
6. Restore the deterministic root cause and action before returning the result.

## Safety boundary

The LLM is **not trusted to calculate or authorize financial outcomes**.

- The deterministic engine decides the root cause.
- The deterministic engine decides the action.
- The deterministic engine decides financial impact.
- Qwen only produces an explanation of those already-computed facts.
- A final safety guard restores the deterministic decision before the result is returned.

If Qwen times out, returns malformed JSON, or otherwise fails to provide usable structured output, the system returns a deterministic evidence explanation instead. The dashboard explicitly labels whether the displayed explanation came from Qwen or the deterministic fallback.

## Streamlit dashboard

`app.py` provides an operations-style dashboard with:

- settlement and bank reconciliation KPIs
- exception explorer
- deterministic finding details
- observable issue codes
- on-demand local Qwen investigation
- confidence and tool-call metrics
- explanation-source labeling
- ground-truth benchmark metrics
- local Qwen consistency metrics
- settlement filtering and overview
- a refresh control for newly generated local results

The UI is intentionally careful not to imply that the model independently made the financial decision.

## Benchmark results

The current recorded deterministic benchmark reports:

- **666/666** corrupted settlements detected
- **0** missed corrupted settlements
- **1** false positive
- **100%** corruption recall
- **99.9%** detection precision
- **99.9%** detection F1
- **99.7%** ground-truth root-cause accuracy

The latest local Qwen consistency run reported:

- **50** cases evaluated
- **100%** root-cause agreement
- **100%** action agreement
- **0** disagreements
- **58%** fallback rate in the recorded run

The fallback rate is expected to vary with local model output quality. It does not weaken the financial safety boundary because fallback preserves the deterministic decision.

## Local setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama and pull the configured local model:

```bash
ollama pull qwen3:0.6b
```

Environment defaults:

```text
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:0.6b
```

## Running locally

The project expects the generated dataset under `data/` and reconciliation outputs under `results/`. These directories are intentionally ignored by Git.

Run deterministic reconciliation:

```bash
python src\\reconcile.py
```

Run the local Qwen sample benchmark:

```bash
python src\\run_ollama_agent.py --limit 50
python src\\evaluate_ollama.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

Use **Refresh local results** in the dashboard after rerunning reconciliation or Ollama evaluation scripts.

Run the automated tests:

```bash
pytest -q
```

## Repository hygiene

Generated datasets and local outputs are deliberately excluded from source control:

```text
data/
results/
src/__pycache__/
.venv/
```

The repository CI compiles Python sources, runs the unit tests, and verifies that generated artifacts are not accidentally committed.

## Project structure

```text
Razorpay/
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Python dependencies
├── tests/                         # Lightweight deterministic/AI safety tests
├── src/
│   ├── finance_tools.py           # Data access / financial tools
│   ├── investigate_exception.py   # Deterministic investigation
│   ├── ai_controller.py           # Evidence packet + decision boundary
│   ├── ollama_agent.py            # Local Qwen explanation layer
│   ├── run_ollama_agent.py        # Batch local inference
│   └── evaluate_ollama.py         # Reconciliation + AI benchmark
└── .github/workflows/ci.yml       # GitHub Actions CI
```

## Limitations

- The dataset is synthetic and is not production Razorpay data.
- The tiny local model can frequently fall back to deterministic evidence wording.
- Benchmark artifacts depend on a working local Python/Ollama environment.
- The Streamlit app reads generated local results rather than orchestrating the complete data pipeline itself.
- Some bank-side corruption categories intentionally collapse to the same operational root cause because the system prioritizes actionable review paths over an unnecessarily granular taxonomy.

## Future improvements

- Add more deterministic tests for edge-case bank reference conflicts.
- Improve compact evidence formatting to reduce tiny-model fallback frequency.
- Add richer payment-level drill-down in the dashboard.
- Track benchmark history across repeated local runs.
- Introduce typed result models for stronger validation across scripts.

## Key design takeaway

The project deliberately uses **deterministic logic for financial truth and an LLM for explanation**. The model can make investigation easier to understand, but it cannot silently change what the reconciliation engine decided.
