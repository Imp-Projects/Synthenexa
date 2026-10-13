# Phase 8 Controlled Experiment Protocol

This protocol is the execution checklist for the limited-budget validation of
the prompt-refinement prototype. It uses Ollama models only and separates the
cheap tuning run from the three-model confirmation run.

## 1. Preflight (no Ollama calls)

```bash
cd /workspaces/Synthenexa
source .venv/bin/activate
pytest -q
```

Confirm that the dataset, seed, cache directory, and artifact directory are
available. Do not delete `synthenexa_pipeline/experiments/cache`; it prevents
repeating identical generation requests.

## 2. Two-model tuning run

Use three generations and three mutations. The call guard is intentionally set
below the available monthly budget:

```bash
cd /workspaces/Synthenexa/synthenexa_pipeline
python scripts/run_optimization.py \
  --ensemble \
  --models "nemotron-3-ultra:cloud,gpt-oss:120b-cloud" \
  --generations 3 \
  --mutations 3 \
  --policy greedy \
  --ensemble-strategy robust \
  --dataset Dataset/benchmark_mbpp_micro.jsonl \
  --seed 0 \
  --max-llm-calls 500 \
  --storage file \
  --experiment phase8_tuning
```

The run must finish without exceeding the call guard. Inspect the manifest
and confirm that persistent cache hits are recorded.

## 3. Offline reporting

```bash
python Analysis/generate_analysis.py --experiment phase8_tuning --source artifacts
python scripts/generate_final_report.py --experiment phase8_tuning
python Analysis/generate_impact_dashboard.py --experiment phase8_tuning
```

Proceed only if validation improves and neither tuning model shows a material
held-out regression. Reports must show the same held-out task IDs for base and
refined prompts.

## 4. Three-model confirmation run

Run this only after the tuning result passes the previous checks:

```bash
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
  --experiment phase8_confirmation
```

## Evidence to record

- Dataset hash and task IDs for train, validation, and test splits.
- Model names, generation/mutation counts, seed, and robust strategy.
- Estimated and actual Ollama calls, cache hits, and failures.
- Baseline/refined search, validation, and held-out scores.
- Per-model held-out deltas and weakest-model delta.
- Whether the improvement is consistent across models, not just the mean.

Do not claim a general 30% improvement from one small split. Treat the result
as prototype feasibility evidence and report the task count prominently.
