# Synthenexa Essential Commands

Synthenexa is currently operated as an Ollama-only prototype. The normal workflow is:

```text
run optimization → generate analysis → generate final report → inspect dashboard
```

Run commands from the repository root unless noted otherwise.

## Setup and tests

```bash
source .venv/bin/activate
pytest -q
```

The full test suite does not contact Ollama.

## 1. Run the Ollama ensemble

Use an explicit model list so the experiment is reproducible and does not accidentally select unsupported providers.

```bash
cd /workspaces/Synthenexa/synthenexa_pipeline
python scripts/run_optimization.py \
  --ensemble \
  --models "nemotron-3-ultra:cloud,gpt-oss:120b-cloud,neeoggl/glm-5.2:latest" \
  --generations 3 \
  --mutations 3 \
  --policy greedy \
  --ensemble-strategy robust \
  --dataset Dataset/benchmark_mbpp_micro.jsonl \
  --seed 0 \
  --max-llm-calls 1000 \
  --storage file \
  --experiment <experiment_name>
```

For the budget-conscious Phase 8 tuning run, use only
`nemotron-3-ultra:cloud,gpt-oss:120b-cloud` and set
`--max-llm-calls 500`. Confirm the validation result before adding
`neeoggl/glm-5.2:latest` for the three-model confirmation run.

The prototype intentionally limits runs to a maximum of three generations and five mutations per generation.

For a cheaper two-model run, omit the third model from `--models`.

## 2. Generate structured analysis

```bash
cd /workspaces/Synthenexa/synthenexa_pipeline
python Analysis/generate_analysis.py --experiment <experiment_name> --source artifacts
```

This creates organized charts and `summary.json`, including reward progression, cross-model robustness, model comparisons, error summaries, and the strict held-out comparison.

## 3. Generate the final report assets

```bash
cd /workspaces/Synthenexa/synthenexa_pipeline
python scripts/generate_final_report.py --experiment <experiment_name>
```

This creates `consolidated_comparison.png`, `improvement_summary.png`, and `comparison_table.md`.

## 4. Generate the four-panel impact dashboard

The recommended artifact-native dashboard reads the `summary.json` produced by the previous step. It uses the strict held-out comparison and performs no database import or live model call:

```bash
cd /workspaces/Synthenexa/synthenexa_pipeline
python Analysis/generate_impact_dashboard.py --experiment <experiment_name>
```

This creates `Analysis/output/<experiment_name>/impact_dashboard.png`. Keep `compare_baseline.py` only for legacy SQLite-backed experiments.

## 5. Run the FastAPI results viewer

```bash
cd /workspaces/Synthenexa
PYTHONPATH=/workspaces/Synthenexa/synthenexa_pipeline \
uvicorn synthenexa_pipeline.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/` in a browser.

## Optional inspection

```bash
cd /workspaces/Synthenexa/synthenexa_pipeline
python scripts/inspect_experiments.py --storage file list
```

The following remain internal or optional:

- `Analysis/visualize_experiments.py` — legacy SQLite visualizations.
- `Analysis/import_artifacts.py` — SQLite compatibility/import support.
- `scripts/run_final_benchmark.py` — repeated benchmark aggregation.
- `scripts/inspect_experiments.py` — optional diagnostics.
