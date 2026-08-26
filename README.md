# Razorpay — Financial Reconciliation AI

A local-first financial reconciliation system that combines deterministic accounting checks with a small local Qwen model for evidence-based exception explanations.

## Why this project

Financial reconciliation should be **correct before it is intelligent**. This project therefore separates the financial decision from the language model:

- The deterministic engine calculates reconciliation outcomes and financial impact.
- A compact evidence packet is sent to local Ollama/Qwen for a human-readable explanation.
- The model cannot override the deterministic root cause or recommended action.
- Invalid model output falls back to a deterministic explanation.

This makes the AI layer useful without putting financial correctness behind an LLM.

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
- Produces a Streamlit dashboard for exception investigation.
- Benchmarks the deterministic engine against controlled corruption.
- Benchmarks Qwen for consistency with deterministic decisions.

## Architecture

```text
CSV data
   │
   ▼
Deterministic reconciliation engine
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

Final decision = deterministic engine
LLM = explanation layer only
```

## Local setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Install Ollama and pull the configured local model:

```bash
ollama pull qwen3:0.6b
```

The agent defaults to:

```text
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:0.6b
```

## Run locally

The project expects the generated dataset under `data/` and reconciliation outputs under `results/`. These directories are intentionally ignored by Git.

Run the local Qwen layer on a sample of exceptions:

```bash
py src\\run_ollama_agent.py --limit 50
```

Evaluate the deterministic engine and Qwen consistency layer:

```bash
py src\\evaluate_ollama.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

Use **Refresh local results** in the dashboard after rerunning the evaluation scripts.

## Benchmarks

The controlled corruption benchmark currently reports:

- **666/666** corrupted settlements detected
- **0** missed corrupted settlements
- **1** false positive
- **100%** corruption recall
- **99.9%** detection precision
- **99.9%** detection F1
- **99.7%** ground-truth root-cause accuracy

The corruption suite contains 666 controlled cases across settlement-item, fee, amount, bank-transaction, partial-settlement, and variance failure modes.

The local Qwen consistency benchmark evaluates whether the explanation layer preserves deterministic decisions. The latest local run achieved **100% root-cause agreement** and **100% action agreement** across 50 evaluated cases. Qwen fallback rate is reported from actual `explanation_source` values and can vary between local runs depending on model output quality.

## Safety boundary

The LLM is **not trusted to calculate or authorize financial outcomes**. It receives deterministic evidence and the already-computed root cause/action. A final safety guard restores those deterministic values before returning the decision.

If Qwen returns malformed or otherwise unusable structured output, the system uses a deterministic evidence explanation instead. The dashboard exposes whether the displayed explanation came from Qwen or the deterministic fallback.

## Repository hygiene

Generated datasets and outputs are deliberately excluded from source control:

```text
data/
results/
src/__pycache__/
```

The repository includes lightweight GitHub Actions CI that compiles the Python sources and checks that generated artifacts are not accidentally committed.

## Project structure

```text
Razorpay/
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Python dependencies
├── src/
│   ├── finance_tools.py           # Data access / financial tools
│   ├── investigate_exception.py   # Deterministic investigation
│   ├── ai_controller.py           # Evidence packet + decision boundary
│   ├── ollama_agent.py            # Local Qwen explanation layer
│   ├── run_ollama_agent.py        # Batch local inference
│   └── evaluate_ollama.py         # Reconciliation + AI benchmark
└── .github/workflows/ci.yml       # Lightweight repository CI
```

## Key design takeaway

The project deliberately uses **deterministic logic for financial truth and an LLM for explanation**. That separation is the core engineering decision: the model can make the investigation easier to understand, but it cannot silently change what the reconciliation engine decided.
