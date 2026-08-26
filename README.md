# Razorpay — Financial Reconciliation AI

A local-first financial reconciliation system that combines deterministic accounting checks with a small local Qwen model for evidence-based exception explanations.

## What it does

- Reconciles settlements, settlement items, payments, and bank transactions.
- Detects settlement and bank-side exceptions such as amount mismatches, missing/duplicate items, fee mismatches, missing bank credits, and reference conflicts.
- Uses the deterministic reconciliation engine as the **financial source of truth**.
- Sends only a compact evidence packet to local Ollama/Qwen for a concise explanation.
- Prevents the LLM from changing the deterministic root cause or recommended action.
- Includes a Streamlit dashboard for exception exploration and benchmark results.

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

Create/activate a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Install and start Ollama, then make sure the configured model is available:

```bash
ollama pull qwen3:0.6b
```

The agent defaults to:

```text
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:0.6b
```

## Run the reconciliation pipeline

The project expects the local generated dataset under `data/` and reconciliation outputs under `results/`.

Run the local Qwen layer on a sample of exceptions:

```bash
py src\\run_ollama_agent.py --limit 50
```

Evaluate both the deterministic engine and the Qwen consistency layer:

```bash
py src\\evaluate_ollama.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

## Benchmarks

The controlled corruption benchmark currently reports:

- 666/666 corrupted settlements detected
- 0 missed corrupted settlements
- 1 false positive
- 100% corruption recall
- 99.9% detection precision
- 99.9% detection F1
- 99.7% ground-truth root-cause accuracy

The local Qwen consistency benchmark evaluates whether the explanation layer preserves deterministic decisions. The model is deliberately not allowed to override the financial decision.

## Safety boundary

The LLM is not trusted to calculate or authorize financial outcomes. It receives deterministic evidence and the already-computed root cause/action. A final safety guard restores those deterministic values before returning the decision. If the local model cannot produce valid structured output, the system falls back to a deterministic evidence explanation.

## Generated data

The `data/`, `results/`, and Python cache directories are local/generated artifacts and should not be committed to the repository. The benchmark can therefore be reproduced locally without making the generated dataset part of the source-code history.
